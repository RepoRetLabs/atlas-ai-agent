#!/usr/bin/env python3
"""
scripts/model_server_manager.py
Atlas AI – Dynamic MLX Model Server Manager
Fixed: state locking, unload races, port reuse, inference protection, concurrency safety.
"""

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
import fcntl
import socket

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
        
        self.port_pool = net_cfg["port_pool"]
        self.port_cycle = cycle(self.port_pool)
        self.max_concurrent = net_cfg["max_concurrent"]
        self.ram_margin_gb = net_cfg["ram_safety_margin_gb"]
        self.large_threshold_gb = net_cfg.get("large_model_ram_threshold_gb", 10)
        self.idle_timeout_min = net_cfg.get("idle_timeout_min", 30)
        
        self.active = self._load_state()  # route → {"port": int, "pid": int, "last_used": float}
        
        self._start_idle_checker()

    def _load_state(self):
        if not STATE_FILE.exists():
            return {}
        try:
            with open(STATE_FILE, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = yaml.safe_load(f) or {}
                fcntl.flock(f, fcntl.LOCK_UN)
            return data
        except Exception:
            return {}

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(STATE_FILE, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                yaml.dump(self.active, f, sort_keys=False)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logging.error(f"Failed to save state: {e}")

    def get_free_ram_gb(self):
        return psutil.virtual_memory().available / (1024 ** 3)

    def _port_is_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _is_active_inference(self, route: str) -> bool:
        """Protect model during active generation (120s window)."""
        if route not in self.active:
            return False
        return time.time() - self.active[route].get("last_used", 0) < 120

    def _unload_route(self, route: str):
        if route not in self.active or self._is_active_inference(route):
            return
        info = self.active[route]
        logging.info(f"Unloading {route} (pid {info['pid']})")
        try:
            os.kill(info["pid"], 15)  # SIGTERM
            for _ in range(15):
                time.sleep(0.5)
                if not psutil.pid_exists(info["pid"]):
                    break
            if psutil.pid_exists(info["pid"]):
                os.kill(info["pid"], 9)
            # Wait for port free
            for _ in range(25):
                if self._port_is_free(info["port"]):
                    break
                time.sleep(0.25)
        except (ProcessLookupError, OSError):
            pass
        except Exception as e:
            logging.error(f"Unload error {route}: {e}")

        if route in self.active:
            del self.active[route]
            self._save_state()

    def _start_idle_checker(self):
        def checker():
            while True:
                time.sleep(60)
                now = time.time()
                to_unload = [
                    r for r, info in list(self.active.items())
                    if now - info.get("last_used", 0) > self.idle_timeout_min * 60
                    and not self._is_active_inference(r)
                ]
                for route in to_unload:
                    logging.info(f"Idle timeout → unloading {route}")
                    self._unload_route(route)
        
        thread = threading.Thread(target=checker, daemon=True, name="IdleChecker")
        thread.start()
        logging.info("Idle timeout checker started")

    def start_server(self, route: str) -> int | None:
        if len(self.active) >= self.max_concurrent:
            oldest = min(self.active, key=lambda r: self.active[r].get("last_used", 0))
            self._unload_route(oldest)

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
            logging.warning(f"Insufficient RAM for {route} ({free:.1f}GB free)")
            return None
        
        # Safe free port from pool
        port = next(self.port_cycle)
        attempts = 0
        max_attempts = len(self.port_pool) * 2
        while not self._port_is_free(port) and attempts < max_attempts:
            port = next(self.port_cycle)
            attempts += 1
        if not self._port_is_free(port):
            logging.error(f"No free port for {route}")
            return None

        log_file = LOGS_DIR / f"server_{route}.log"
        cmd = ["mlx_lm.server", "--model", str(model_path), "--port", str(port)]
        proc = subprocess.Popen(cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
        pid = proc.pid

        # Health check
        url = f"http://127.0.0.1:{port}/health"
        for _ in range(18):
            time.sleep(2)
            try:
                r = httpx.get(url, timeout=3)
                if r.status_code == 200:
                    logging.info(f"{route} ready on port {port} (pid {pid})")
                    break
            except:
                pass
        else:
            logging.error(f"{route} failed to start")
            try:
                os.kill(pid, 9)
            except:
                pass
            return None
        
        self.active[route] = {"port": port, "pid": pid, "last_used": time.time()}
        self._save_state()
        return port

    def get_port(self, route: str) -> int | None:
        if route in self.active:
            if psutil.pid_exists(self.active[route]["pid"]):
                self.active[route]["last_used"] = time.time()
                self._save_state()
                return self.active[route]["port"]
            else:
                logging.info(f"Dead process for {route} – cleaning")
                del self.active[route]
                self._save_state()
        
        return self.start_server(route)


if __name__ == "__main__":
    mgr = ModelServerManager()
    port = mgr.get_port("general_fast")
    print(f"Port for general_fast: {port}")