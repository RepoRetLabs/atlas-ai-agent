#!/usr/bin/env python3
"""
scripts/proxy.py
Atlas AI – FINAL proxy: strong few-shot + length enforcement + aggressive expansion + line filter
"""

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
STATE_DIR = PROJECT_ROOT / "state"
CONFIG_DIR = PROJECT_ROOT / "configs"
LOG_FILE = LOGS_DIR / "proxy.log"
ROUTER_URL = "http://127.0.0.1:8083/v1/chat/completions"
STATE_DIR.mkdir(parents=True, exist_ok=True)

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

app = FastAPI(title="Atlas Proxy")
mgr = ModelServerManager()

xml_path = CONFIG_DIR / "models_router.xml"
tree = ET.parse(xml_path)
ROUTES_XML = ET.tostring(tree.getroot(), encoding='unicode', method='xml')
AVAILABLE_ROUTES = [route.attrib['name'] for route in tree.getroot().findall("route")]

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512

def load_history_cache(route: str):
    cache_file = STATE_DIR / f"last_{route}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except:
            return []
    return []

def save_history_cache(route: str, messages: list):
    cache_file = STATE_DIR / f"last_{route}.json"
    with open(cache_file, "w") as f:
        json.dump(messages[-10:], f)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    request_id = str(uuid.uuid4())[:8]
    extra = {"request_id": request_id}

    last_user = req.messages[-1]["content"].lower()

    if req.model:
        route = req.model
    elif any(k in last_user for k in ["scary", "story", "horror", "write a", "creative", "ideation"]):
        route = "creative_explore"
    else:
        route_prompt = f'Output ONLY: {{"route":"exact_route_name"}}\nRoutes:<routes>{ROUTES_XML}</routes>\nConv:{json.dumps(req.messages)}'
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(ROUTER_URL, json={"messages":[{"role":"user","content":route_prompt}],"temperature":0.0,"max_tokens":64}, timeout=10)
                content = re.sub(r'<\|.*?\|>|<think>.*?</think>', '', r.json()["choices"][0]["message"]["content"])
                route = json.loads(content).get("route", "general_fast")
        except:
            route = "general_fast"
    if route not in AVAILABLE_ROUTES:
        route = "general_fast"

    port = mgr.get_port(route)
    if not port:
        route = "general_fast"
        port = mgr.get_port(route)

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = req.model_dump(exclude={"model"})
    payload["repetition_penalty"] = 1.25
    payload["stop"] = ["<|end|>"]

    is_creative = route == "creative_explore"

    if is_creative:
        payload["max_tokens"] = 8192
        payload["temperature"] = 0.92
        creative_system = {"role": "system", "content": """You are a master storyteller. ALWAYS reply EXACTLY in this format and NOTHING ELSE:

<think>One short sentence</think>

**Story:**
Full 350-450 word scary story. Pure narrative. Start immediately. Never stop early. No planning.

Examples:
<think>Haunted house twist</think>

**Story:**
The old door creaked open in the dead of night. A cold wind rushed past me as I stepped into the abandoned Victorian mansion on Willow Street. The floorboards groaned under my weight like living bones. Dust danced in the moonlight streaming through cracked windows. Suddenly, a whisper echoed from the darkness: "Welcome home..." I froze. The voice was my own.

<think>Forest ghost idea</think>

**Story:**
Deep in the ancient woods, the trail vanished. I had been hiking for hours when the trees seemed to close in. A pale figure stood between two massive oaks, its eyes glowing like dying embers. It reached out a hand that was not quite human. I ran, but every path led back to it. The figure smiled with my mother's face..."""}
        payload["messages"] = [creative_system] + payload["messages"]
    else:
        payload["max_tokens"] = max(1024, req.max_tokens)
        payload["temperature"] = 0.7
        payload["messages"] = [{"role": "system", "content": "Direct response only."}] + payload["messages"]

    cache = load_history_cache(route)
    if cache:
        payload["messages"] = cache + payload["messages"][-4:]
    try:
        query_embedding = init_memory.embedder.encode(req.messages[-1]["content"]).tolist()
        results = init_memory.collection.query(query_embeddings=[query_embedding], n_results=3, include=["documents", "metadatas", "distances"])
        memories = [doc for doc, dist in zip(results["documents"][0], results["distances"][0]) if dist < 0.45]
        if memories:
            payload["messages"] = [{"role": "system", "content": f"Memory:\n" + "\n".join(memories)}] + payload["messages"]
    except:
        pass

    logger.info(f"Forwarding to {route} on port {port}", extra=extra)

    async with httpx.AsyncClient() as client:
        response_text = ""
        for attempt in range(6):
            resp = await client.post(url, json=payload, timeout=300)
            data = resp.json()
            temp_text = data["choices"][0]["message"]["content"].strip()

            if is_creative:
                match = re.search(r'\*\*Story:\*\*(.*)', temp_text, re.DOTALL | re.IGNORECASE)
                story_part = match.group(1).strip() if match else temp_text
                if len(story_part) < 250:
                    logger.warning(f"Short story (attempt {attempt+1}, {len(story_part)} chars) - expanding", extra=extra)
                    payload["messages"].append({"role": "user", "content": "This is too short. Write a COMPLETE 400-word scary story now. Continue from the last sentence in exact format."})
                    payload["max_tokens"] += 1024
                    continue

            if len(temp_text) > 200:
                response_text = temp_text
                break

            payload["messages"].append({"role": "user", "content": "Continue exact format." if is_creative else "Complete."})
            payload["max_tokens"] += 1024

        if is_creative:
            match = re.search(r'\*\*Story:\*\*(.*)', response_text, re.DOTALL | re.IGNORECASE)
            story = match.group(1).strip() if match else response_text

            # Aggressive line filter - remove any line with planning words
            bad_words = ["lets", "we must", "produce", "count", "length", "words", "proceed", "we need", "thinking", "write about", "example", "narrative", "meta", "planning", "we'll"]
            story_lines = [line for line in story.split('\n') if not any(bad in line.lower() for bad in bad_words)]
            story = '\n'.join(story_lines).strip()

            think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)
            think = think_match.group(1).strip() if think_match else "Story generated"

            response_text = f"<think>{think}</think>\n\n**Story:**\n{story}"

        response_text = response_text.strip() or "Sorry, response generation failed. Please try again."

        data["model"] = route
        data["choices"][0]["message"]["content"] = response_text

        save_history_cache(route, payload["messages"] + [{"role": "assistant", "content": response_text}])
        try:
            response_embedding = init_memory.embedder.encode(response_text).tolist()
            init_memory.collection.add(documents=[response_text], embeddings=[response_embedding], ids=[request_id], metadatas=[{"user_id": "system", "source": "response"}])
        except:
            pass

        logger.info(f"Success | prompt:{data.get('usage',{}).get('prompt_tokens')} completion:{data.get('usage',{}).get('completion_tokens')}", extra=extra)
        return data

if __name__ == "__main__":
    logger.info("Starting uvicorn on port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")