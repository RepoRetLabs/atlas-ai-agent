#!/usr/bin/env python3
"""
Atlas AI Agent – start.py
One-time / repeatable setup + service start + interactive prompt
Uses configs/model_description.xml as source of truth for routes & models
"""

import os
import sys
import subprocess
import time
import logging
import socket
from pathlib import Path
import requests
import yaml
import xml.etree.ElementTree as ET
import json
import psutil
import signal

os.environ["HF_OFFLINE_MODE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent

LOGS_DIR    = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MEMORY_DIR  = PROJECT_ROOT / "memory" / "chroma_db"
STATE_DIR   = PROJECT_ROOT / "state"
STATE_FILE  = STATE_DIR / "active_models.yaml"

ROUTER_PORT = 8083
ROUTER_MODEL = "models/Arch-Router-1.5B-mlx"
ROUTER_URL  = f"http://127.0.0.1:{ROUTER_PORT}/health"

PROXY_PORT = 8000
PROXY_URL  = f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions"

DEFAULT_SYSTEM = "You are Atlas, a helpful local AI agent."

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[logging.FileHandler(LOGS_DIR / "start.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("atlas-start")

# ───────────────────────────────────────────────
# Startup Cleanup
# ───────────────────────────────────────────────

def kill_stale_processes(patterns):
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            name = proc.info['name'] or ''
            if any(pat in cmd or pat in name for pat in patterns):
                logger.info(f"Terminating stale {proc.pid}  {name}  {cmd[:60]}...")
                proc.send_signal(signal.SIGTERM)
                time.sleep(1)
                if proc.is_running():
                    proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed

def cleanup_startup():
    logger.info("Performing startup cleanup...")
    mlx_patterns = ["mlx_lm.server", "mlx_lm generate"]
    proxy_patterns = ["python.*proxy.py", "uvicorn"]
    killed_mlx = kill_stale_processes(mlx_patterns)
    killed_proxy = kill_stale_processes(proxy_patterns)
    logger.info(f"Killed {killed_mlx} mlx + {killed_proxy} proxy processes")

    if STATE_FILE.exists():
        STATE_FILE.unlink()
        logger.info("Cleared active_models.yaml")

    log_patterns = ["server_", "proxy.log", "router.log", "model_manager.log"]
    cleared = 0
    for item in LOGS_DIR.iterdir():
        if item.is_file() and any(p in item.name for p in log_patterns):
            item.unlink()
            cleared += 1
    logger.info(f"Cleared {cleared} dynamic logs")

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────

def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

def ask_atlas(messages, model: str | None = None, temperature: float = 0.7, max_tokens: int = 512) -> str:
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.2,
    }
    if model:
        payload["model"] = model

    try:
        resp = requests.post(PROXY_URL, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error contacting proxy: {str(e)}"

# ───────────────────────────────────────────────
# Load model description from XML (source of truth)
# ───────────────────────────────────────────────

def load_model_description_xml():
    xml_path = CONFIGS_DIR / "model_description.xml"
    if not xml_path.exists():
        logger.error(f"Missing source file: {xml_path}")
        sys.exit(1)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    model_data = {}
    for route_elem in root.findall("route"):
        name = route_elem.get("name")
        folder = route_elem.find("folder").text.strip() if route_elem.find("folder") is not None else None
        ram = route_elem.find("ram_gb_approx")
        desc = route_elem.find("description")

        if not name or not folder:
            logger.warning(f"Invalid route entry in XML (missing name or folder)")
            continue

        model_data[folder] = {
            "routes": [name],
            "ram_gb_approx": int(ram.text.strip()) if ram is not None and ram.text else 8,
            "description": desc.text.strip() if desc is not None and desc.text else "No description"
        }

    return model_data

# ───────────────────────────────────────────────
# Generate registry and router files from XML
# ───────────────────────────────────────────────

def generate_routing_files():
    model_data = load_model_description_xml()
    models_dir = PROJECT_ROOT / "models"
    available = {d.name for d in models_dir.iterdir() if d.is_dir()}

    registry = {"routes": {}}
    for folder, info in model_data.items():
        if folder in available:
            for route in info["routes"]:
                registry["routes"][route] = {
                    "folder": folder,
                    "ram_gb_approx": info["ram_gb_approx"]
                }
        else:
            logger.warning(f"Model folder not found: {folder} → skipping route(s) {info['routes']}")

    # Write models_registry.yaml
    yaml_path = CONFIGS_DIR / "models_registry.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(registry, f, sort_keys=False)
    logger.info(f"Generated models_registry.yaml ({len(registry['routes'])} routes)")

    # Write models_router.xml (only available routes)
    xml_content = '<routes>\n'
    for route, info in registry["routes"].items():
        desc = model_data[info["folder"]]["description"]
        xml_content += f'  <route name="{route}" description="{desc}"/>\n'
    xml_content += '</routes>'

    router_xml_path = CONFIGS_DIR / "models_router.xml"
    with open(router_xml_path, "w") as f:
        f.write(xml_content)
    logger.info(f"Generated models_router.xml ({len(registry['routes'])} routes)")

# ───────────────────────────────────────────────
# Other setup functions (unchanged core logic)
# ───────────────────────────────────────────────

def create_directories():
    dirs = ["models", "configs", "scripts", "scripts/utils", "state", "memory/chroma_db", "logs"]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

def write_default_configs():
    network = """
router:
  port: 8083
  model_path: models/Arch-Router-1.5B-mlx

model_server:
  port_pool: [8084, 8085, 8086, 8087, 8088]
  max_concurrent: 2
  ram_safety_margin_gb: 4
  large_model_ram_threshold_gb: 10
  idle_timeout_min: 10
"""
    with open(CONFIGS_DIR / "network.yaml", "w") as f:
        f.write(network)

    memory = """
llm:
  base_url: http://127.0.0.1:8083
  api_key: "not-needed"
  model: "general_fast"
"""
    with open(CONFIGS_DIR / "memory.yaml", "w") as f:
        f.write(memory)

def init_memory():
    subprocess.run(["python", str(PROJECT_ROOT / "scripts" / "init_memory.py")], check=False)

def start_router():
    if not is_port_free(ROUTER_PORT):
        kill_stale_processes(["mlx_lm.server"])
    log_file = LOGS_DIR / "router.log"
    cmd = ["nohup", "mlx_lm.server", "--model", str(PROJECT_ROOT / ROUTER_MODEL), "--port", str(ROUTER_PORT)]
    proc = subprocess.Popen(cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    for _ in range(45):
        time.sleep(2)
        try:
            if requests.get(ROUTER_URL, timeout=3).ok:
                return
        except:
            pass
    sys.exit(1)

def start_proxy():
    if not is_port_free(PROXY_PORT):
        kill_stale_processes(["proxy.py", "uvicorn"])
    log_file = LOGS_DIR / "proxy.log"
    cmd = ["nohup", "python", str(PROJECT_ROOT / "scripts" / "proxy.py")]
    subprocess.Popen(cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    for _ in range(20):
        time.sleep(2)
        try:
            if requests.get(f"http://127.0.0.1:{PROXY_PORT}/docs", timeout=5).status_code == 200:
                return
        except:
            pass
    sys.exit(1)

# ───────────────────────────────────────────────
# Interactive prompt (unchanged from last working version)
# ───────────────────────────────────────────────

def load_available_routes():
    xml_path = CONFIGS_DIR / "models_router.xml"
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    return [{"name": r.attrib["name"], "desc": r.attrib["description"]}
            for r in tree.getroot().findall("route")]

def print_welcome_screen(routes):
    print("\n" + "═" * 70)
    print("          ATLAS AI AGENT – Interactive Session")
    print("═" * 70)
    print(f"Proxy: http://127.0.0.1:{PROXY_PORT}\n")
    print("Available routes:")
    for r in sorted(routes, key=lambda x: x['name']):
        desc = r['desc'][:55] + ('...' if len(r['desc']) > 55 else '')
        print(f"  • {r['name']:18}  {desc}")
    print("\nQuick tags: /code  /think|/deep  /ideation  use <route>")
    print("═" * 70 + "\n")

def launch_prompt():
    routes_data = load_available_routes()
    routes = [r['name'] for r in routes_data]
    print_welcome_screen(routes_data)

    history = [{"role": "system", "content": DEFAULT_SYSTEM}]
    last_response = ""

    tag_map = {
        "/ideation": "creative_explore",
        "/critique": "coding_critic",
        "/review":   "coding_critic",
        "/deep":     "reasoning_deep",
        "/think":    "reasoning_deep",
        "/code":     "coding_expert",
        "/fast":     "reasoning_fast",
        "/quick":    "general_fast",
        "/vision":   "vision",
        "/strong":   "general_strong",
        "/grok":     "grok_api"
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                if last_response: print("\nAtlas:", last_response)
                continue

            model_override = None
            content = user_input

            if user_input.lower().startswith("use "):
                parts = user_input.split(maxsplit=2)
                if len(parts) >= 2 and parts[1].strip() in routes:
                    model_override = parts[1].strip()
                    content = parts[2].strip() if len(parts) > 2 else ""

            elif user_input.startswith("/"):
                tag = user_input.split(maxsplit=1)[0]
                if tag in tag_map:
                    model_override = tag_map[tag]
                    content = user_input[len(tag):].lstrip()

            if not content:
                content = "(repeat last question)"

            history.append({"role": "user", "content": content})
            response = ask_atlas(history, model=model_override)
            print("\nAtlas:", response)
            history.append({"role": "assistant", "content": response})
            last_response = response

            if len(history) > 20:
                history = history[:1] + history[-19:]

        except KeyboardInterrupt:
            print("\n\nSession ended.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    os.chdir(PROJECT_ROOT)
    cleanup_startup()
    create_directories()
    write_default_configs()
    generate_routing_files()
    init_memory()

    print("""
┌────────────────────────────────────────────────────────────┐
│                 ATLAS AI AGENT – SAFETY NOTICE             │
│                                                            │
│ • Runs with your user permissions                          │
│ • No sandbox or output filtering                           │
│ • You are responsible for all generated content            │
│ • Review logs regularly                                    │
│ • Avoid high-stakes use without verification               │
└────────────────────────────────────────────────────────────┘
""")

    start_router()
    start_proxy()

    logger.info("Services ready. Launching interactive prompt.")
    launch_prompt()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)