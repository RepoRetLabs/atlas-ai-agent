# scripts/proxy.py
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import uuid
import json
import re
import xml.etree.ElementTree as ET
from model_server_manager import ModelServerManager
import init_memory

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs"
LOG_FILE = LOGS_DIR / "proxy.log"
ROUTER_URL = "http://127.0.0.1:8083/v1/chat/completions"

class RequestIDFormatter(logging.Formatter):
    def format(self, record):
        record.request_id = getattr(record, 'request_id', "")
        return super().format(record)

logger = logging.getLogger("atlas_proxy")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
formatter = RequestIDFormatter("%(asctime)s %(levelname)s [req:%(request_id)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logger.info("Proxy logger initialized")

app = FastAPI(title="Atlas Proxy")

mgr = ModelServerManager()
logger.info("ModelServerManager initialized")

# Load routes XML once
xml_path = CONFIG_DIR / "models_router.xml"
try:
    tree = ET.parse(xml_path)
    ROUTES_XML = ET.tostring(tree.getroot(), encoding='unicode', method='xml')
    AVAILABLE_ROUTES = [route.attrib['name'] for route in tree.getroot().findall("route")]
except Exception as e:
    logger.error(f"Failed to load models_router.xml: {e}")
    ROUTES_XML = "<routes></routes>"
    AVAILABLE_ROUTES = []

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    request_id = str(uuid.uuid4())[:8]
    extra = {"request_id": request_id}

    # 1. Route selection
    if req.model:
        route = req.model
        logger.info(f"Explicit route: {route}", extra=extra)
    else:
        route_prompt = f"""
You are a precise route selector. Output ONLY valid JSON.

Use ONLY these routes:

<routes>
{ROUTES_XML}
</routes>

<conversation>
{json.dumps(req.messages, ensure_ascii=False)}
</conversation>

Rules:
- Choose the single best route name based on latest user intent.
- If no strong match or intent already fulfilled → return "other"
- Output EXACTLY this format and nothing else:
{{"route": "exact_route_name"}}
- No reasoning. No extra text. No markdown. No explanations.
- Use double quotes only. Do not use single quotes.
"""

        router_payload = {
            "messages": [{"role": "user", "content": route_prompt}],
            "temperature": 0.0,
            "max_tokens": 64
        }

        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(ROUTER_URL, json=router_payload, timeout=10)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"].strip()
                logger.debug(f"Raw router output: {content}", extra=extra)

                # Robust extraction
                match = re.search(r'\{.*?"route".*?:.*?"([^"]+)"', content, re.DOTALL)
                if match:
                    route = match.group(1)
                else:
                    # Try json.loads on cleaned string
                    cleaned = re.sub(r"['`]", '"', content)
                    try:
                        data = json.loads(cleaned)
                        route = data.get("route", "general_fast")
                    except json.JSONDecodeError as je:
                        logger.error(f"Router JSON parse failed: {je} | raw: {content} → fallback", extra=extra)
                        route = "general_fast"

                logger.info(f"Router selected: {route}", extra=extra)
        except Exception as e:
            logger.error(f"Router call failed: {str(e)} → fallback", extra=extra)
            route = "general_fast"

    # Validate route
    if route not in AVAILABLE_ROUTES and route != "other":
        logger.warning(f"Invalid route {route} → fallback", extra=extra)
        route = "general_fast"

    if route == "other":
        route = "general_fast"

    # 2. Get or start backend
    port = mgr.get_port(route)
    if not port:
        logger.warning(f"Cannot load {route} → fallback to general_fast", extra=extra)
        route = "general_fast"
        port = mgr.get_port(route)
        if not port:
            raise HTTPException(503, "No available model server")

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = req.model_dump(exclude={"model"})
    payload["repetition_penalty"] = 1.2
    payload["stop"] = ["<|end|>"]

    # 3. Memory injection (unchanged)
    query_text = req.messages[-1]["content"]
    try:
        query_embedding = init_memory.embedder.encode(query_text).tolist()
        results = init_memory.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        memories = [doc for doc, dist in zip(results["documents"][0], results["distances"][0]) if dist < 0.45]
        if memories:
            memory_text = "\n".join(memories)
            system_msg = {"role": "system", "content": f"Relevant past memory (use if helpful):\n{memory_text}"}
            payload["messages"] = [system_msg] + payload["messages"]
            logger.info(f"Injected {len(memories)} memories", extra=extra)
    except Exception as e:
        logger.warning(f"Memory search failed: {e}", extra=extra)

    # 4. Forward request
    logger.info(f"Forwarding to {route} on port {port}", extra=extra)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            data["model"] = route

            usage = data.get("usage", {})
            logger.info(
                f"Success | prompt:{usage.get('prompt_tokens')} "
                f"completion:{usage.get('completion_tokens')} total:{usage.get('total_tokens')}",
                extra=extra
            )

            # 5. Save response to memory
            response_text = data["choices"][0]["message"]["content"]
            response_embedding = init_memory.embedder.encode(response_text).tolist()
            init_memory.collection.add(
                documents=[response_text],
                embeddings=[response_embedding],
                ids=[request_id],
                metadatas=[{"user_id": "system", "source": "response"}]
            )
            logger.info("Saved response to memory", extra=extra)

            return data
        except httpx.HTTPError as e:
            logger.error(f"Backend error on {route}: {str(e)}", extra=extra)
            raise HTTPException(502, f"Backend error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting uvicorn on port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")