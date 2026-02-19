#!/usr/bin/env python3
"""
Atlas AI Agent – start.py (ROUTER ONLY + VISIBLE THINKING)
Router now forced to show internal reasoning in <think> tags for diagnosis.
"""

import os
import sys
import subprocess
import time
import logging
import socket
from pathlib import Path
import requests
import json
from datetime import datetime

os.environ["HF_OFFLINE_MODE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent

LOGS_DIR    = PROJECT_ROOT / "logs"
DECISIONS_LOG = LOGS_DIR / "router_decisions.log"

ROUTER_PORT = 8083
ROUTER_MODEL = "models/Arch-Router-1.5B-mlx"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[logging.FileHandler(LOGS_DIR / "start.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("atlas-start")

def kill_stale_processes(patterns):
    import psutil, signal
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            if any(pat in cmd for pat in patterns):
                logger.info(f"Terminating stale {proc.pid}")
                proc.send_signal(signal.SIGTERM)
                time.sleep(1)
                if proc.is_running():
                    proc.kill()
                killed += 1
        except:
            pass
    return killed

def cleanup_startup():
    logger.info("Performing startup cleanup...")
    kill_stale_processes(["mlx_lm.server"])
    logger.info("Cleanup done.")

def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

def start_router():
    if not is_port_free(ROUTER_PORT):
        kill_stale_processes(["mlx_lm.server"])
    log_file = LOGS_DIR / "router.log"
    cmd = ["nohup", "mlx_lm.server", "--model", str(PROJECT_ROOT / ROUTER_MODEL), "--port", str(ROUTER_PORT)]
    subprocess.Popen(cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    for _ in range(30):
        time.sleep(2)
        try:
            if requests.get(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=3).ok:
                logger.info("Router ready on 8083")
                return
        except:
            pass
    logger.error("Router failed to start")
    sys.exit(1)

def log_decision(user_input: str, router_response: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "router_raw_response": router_response,
        "length": len(router_response)
    }
    with open(DECISIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Decision logged: {len(router_response)} chars")

def ask_router(messages):
    try:
        resp = requests.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions", json={
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }, timeout=90)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error contacting router: {str(e)}"

def print_welcome():
    print("\n" + "═" * 70)
    print("          ATLAS AI AGENT – ROUTER ONLY MODE (Visible Thinking)")
    print("═" * 70)
    print("Router now shows internal <think> reasoning for every reply.")
    print("Decisions logged to logs/router_decisions.log")
    print("Type 'exit' to quit.\n")
    print("═" * 70 + "\n")

def launch_prompt():
    print_welcome()
    history = [{
        "role": "system",
        "content": "You are Atlas. ALWAYS show your thinking first in <think>short reasoning</think>, then give the final answer clearly."
    }]

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Session ended.")
                break

            history.append({"role": "user", "content": user_input})
            response = ask_router(history)
            print("\nAtlas:", response)
            history.append({"role": "assistant", "content": response})

            log_decision(user_input, response)

            if len(history) > 20:
                history = history[:1] + history[-15:]

        except KeyboardInterrupt:
            print("\n\nSession ended.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    os.chdir(PROJECT_ROOT)
    cleanup_startup()

    print("Starting Atlas in Router-Only mode with visible internal thinking...")

    start_router()

    logger.info("Router ready. Launching interactive prompt.")
    launch_prompt()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)