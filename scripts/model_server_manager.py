# scripts/model_server_manager.py
import os
import yaml
import subprocess
import time
import psutil
import logging
import httpx
from itertools import cycle
import threading
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = PROJECT_ROOT / "state" / "active_models.yaml"

logging.basicConfig(
    filename=LOGS_DIR / "model_manager.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

class ModelServerManager:
    def __init__(self):
        with open(CONFIG_DIR / "network.yaml") as f:
            net_cfg = yaml.safe_load(f)["model_server"]
        with open(CONFIG_DIR / "models_registry.yaml") as f:
            self.registry = yaml.safe_load(f)["routes"]
        
        self.port_cycle = cycle(net_cfg["port_pool"])
        self.max_concurrent = net_cfg["max_concurrent"]
        self.ram_margin_gb = net_cfg["ram_safety_margin_gb"]
        self.large_threshold_gb = net_cfg.get("large_model_ram_threshold_gb", 10)
        self.idle_timeout_min = net_cfg.get("idle_timeout_min", 10)
        self.active = self._load_state()  # route → {"port": int, "pid": int, "last_used": float}
        
        self._start_idle_checker()

    def _load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            yaml.dump(self.active, f)

    def get_free_ram_gb(self):
        return psutil.virtual_memory().available / (1024 ** 3)

    def _unload_route(self, route):
        if route not in self.active:
            return
        pid = self.active[route]["pid"]
        logging.info(f"Unloading {route} (pid {pid})")
        try:
            os.kill(pid, 15)  # SIGTERM
            time.sleep(5)
            if psutil.pid_exists(pid):
                logging.warning(f"Force killing {route}")
                os.kill(pid, 9)  # SIGKILL
        except ProcessLookupError:
            pass
        del self.active[route]
        self._save_state()

    def _start_idle_checker(self):
        def checker():
            while True:
                time.sleep(60)
                now = time.time()
                to_unload = [
                    r for r, info in list(self.active.items())
                    if now - info["last_used"] > self.idle_timeout_min * 60
                ]
                for route in to_unload:
                    logging.info(f"Idle timeout ({self.idle_timeout_min} min) → unloading {route}")
                    self._unload_route(route)
        
        thread = threading.Thread(target=checker, daemon=True, name="IdleChecker")
        thread.start()
        logging.info("Idle timeout checker thread started")

    def start_server(self, route: str) -> int | None:
        if len(self.active) >= self.max_concurrent:
            self._unload_route(min(self.active, key=lambda r: self.active[r]["last_used"]))

        if route not in self.registry:
            logging.error(f"Unknown route: {route}")
            return None
        
        model_path = PROJECT_ROOT / "models" / self.registry[route]["folder"]
        if not model_path.exists():
            logging.error(f"Model path missing: {model_path}")
            return None
        
        ram_needed = self.registry[route]["ram_gb_approx"]
        free = self.get_free_ram_gb()
        if ram_needed >= self.large_threshold_gb and free < ram_needed + self.ram_margin_gb:
            logging.warning(f"Large model {route} ({ram_needed} GB) rejected: only {free:.1f} GB free")
            return None
        
        port = next(self.port_cycle)
        log_file = LOGS_DIR / f"server_{route}.log"
        
        cmd = ["mlx_lm.server", "--model", str(model_path), "--port", str(port)]
        proc = subprocess.Popen(cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
        pid = proc.pid
        
        # Health check
        url = f"http://127.0.0.1:{port}/health"
        for attempt in range(15):
            time.sleep(2)
            try:
                r = httpx.get(url, timeout=2)
                if r.status_code == 200 and "ok" in r.text.lower():
                    logging.info(f"{route} healthy on port {port} (pid {pid})")
                    break
            except:
                pass
        else:
            logging.error(f"{route} failed to become healthy")
            try:
                os.kill(pid, 9)
            except:
                pass
            return None
        
        self.active[route] = {"port": port, "pid": pid, "last_used": time.time()}
        self._save_state()
        logging.info(f"Started {route} on port {port}")
        return port

    def get_port(self, route: str) -> int | None:
        if route in self.active:
            if psutil.pid_exists(self.active[route]["pid"]):
                self.active[route]["last_used"] = time.time()
                self._save_state()
                return self.active[route]["port"]
            else:
                logging.info(f"Detected dead process for {route}")
                del self.active[route]
                self._save_state()
        
        return self.start_server(route)

if __name__ == "__main__":
    mgr = ModelServerManager()
    port = mgr.get_port("general_fast")
    print(f"Port for general_fast: {port}")