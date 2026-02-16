#!/usr/bin/env python3
"""
Atlas AI Agent – setup.py
One-time / repeatable setup script
No elevated privileges required
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
import requests

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

PROJECT_ROOT = Path.home() / "atlas-ai-agent"
LOGS_DIR     = PROJECT_ROOT / "logs"
CONFIGS_DIR  = PROJECT_ROOT / "configs"

ROUTER_PORT  = 8083
ROUTER_MODEL = "models/Arch-Router-1.5B-mlx"
HEALTH_URL   = f"http://127.0.0.1:{ROUTER_PORT}/health"

# ───────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "setup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("atlas-setup")


def is_router_running() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=2.0)
        return r.status_code == 200 and "healthy" in r.text.lower() or "ok" in r.text.lower()
    except requests.RequestException:
        return False


def create_directories():
    dirs = [
        "models",
        "configs",
        "scripts",
        "scripts/utils",
        "state",
        "memory",
        "memory/chroma_db",
        "logs",
    ]
    for d in dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {path.relative_to(PROJECT_ROOT)}")


def write_default_configs():
    # network.yaml
    network_yaml = CONFIGS_DIR / "network.yaml"
    if not network_yaml.exists():
        network_yaml.write_text(f"""\
router:
  port: {ROUTER_PORT}
  model_path: {ROUTER_MODEL}

model_server:
  port_pool: [8084, 8085, 8086, 8087, 8088]
  max_concurrent: 3
  ram_safety_margin_gb: 6
  large_model_ram_threshold_gb: 10
  session_token_cap_large_gb_under: 8
""")
        logger.info(f"Created {network_yaml.name}")

    # memory.yaml
    memory_yaml = CONFIGS_DIR / "memory.yaml"
    if not memory_yaml.exists():
        memory_yaml.write_text("""\
llm:
  base_url: http://127.0.0.1:8083
  api_key: "not-needed"
  model: "general_fast"
""")
        logger.info(f"Created {memory_yaml.name}")

    # models_registry.yaml placeholder
    registry = CONFIGS_DIR / "models_registry.yaml"
    if not registry.exists():
        registry.write_text("""\
# Placeholder — fill from LLM-13-Models-List-Definition.md
routes:
  general_fast:
    folder: Phi-3.5-mini-instruct
    quant: 4bit
    ram_gb_approx: 3.2
""")
        logger.info(f"Created placeholder {registry.name}")


def start_router():
    if is_router_running():
        logger.info(f"Router already healthy on port {ROUTER_PORT}")
        return

    log_file = LOGS_DIR / "router.log"
    cmd = (
        f"nohup mlx_lm.server "
        f"--model {PROJECT_ROOT / ROUTER_MODEL} "
        f"--port {ROUTER_PORT} "
        f"> {log_file} 2>&1 &"
    )

    logger.info("Launching Arch-Router in background...")
    subprocess.Popen(cmd, shell=True)

    logger.info("Waiting for router to become ready (up to ~45s)...")
    total_wait = 0.0
    wait_steps = [4, 4, 4, 5, 5, 6, 8, 10]  # progressive

    for wait in wait_steps:
        time.sleep(wait)
        total_wait += wait
        if is_router_running():
            logger.info(f"Router healthy after {total_wait:.1f}s")
            return
        logger.info(f"... still waiting ({total_wait:.1f}s elapsed)")

    logger.warning(f"Router not healthy after {total_wait}s — check logs/router.log")


def print_safety_notice():
    print("""
┌────────────────────────────────────────────────────────────┐
│                   ATLAS AI AGENT – SAFETY NOTICE          │
│                                                            │
│ • Runs with your user permissions                          │
│ • No sandbox or output filtering                           │
│ • You are responsible for all generated content            │
│ • Review logs regularly                                    │
│ • Avoid high-stakes use without verification               │
└────────────────────────────────────────────────────────────┘
""")
    logger.info("Safety notice displayed")


def main():
    logger.info("Starting Atlas AI Agent setup...")
    os.chdir(PROJECT_ROOT)

    create_directories()
    write_default_configs()
    print_safety_notice()
    start_router()

    logger.info("Setup completed.")
    logger.info("Next:")
    logger.info("  • Verify: curl http://127.0.0.1:8083/health")
    logger.info("  • Test:   python scripts/prompt.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Setup interrupted")
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        sys.exit(1)