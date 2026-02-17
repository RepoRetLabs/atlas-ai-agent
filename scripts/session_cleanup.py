#!/usr/bin/env python3
"""
scripts/session_cleanup.py

Cleans up:
- All running mlx_lm.server processes
- Proxy/router related processes (if identifiable)
- Active model state file
- All dynamically generated log files
- Temporary / stale files in logs/, state/, memory/ (if applicable)

Run from project root: python scripts/session_cleanup.py
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path
import psutil
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR     = PROJECT_ROOT / "logs"
STATE_DIR    = PROJECT_ROOT / "state"
MEMORY_DIR   = PROJECT_ROOT / "memory" / "chroma_db"

STATE_FILE   = STATE_DIR / "active_models.yaml"

def kill_processes_by_name(name_patterns):
    """Kill processes matching any of the name patterns."""
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            name = proc.info['name'] or ''
            if any(pat in cmd or pat in name for pat in name_patterns):
                print(f"Terminating {proc.pid}  {name}  {cmd[:80]}...")
                proc.send_signal(signal.SIGTERM)
                time.sleep(0.8)
                if proc.is_running():
                    print(f"  → force kill {proc.pid}")
                    proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return killed

def clear_file(path: Path):
    if path.exists():
        try:
            path.unlink()
            print(f"Removed: {path}")
        except Exception as e:
            print(f"Failed to remove {path}: {e}")

def clear_directory_contents(directory: Path, patterns=None):
    if not directory.exists():
        return
    count = 0
    for item in directory.iterdir():
        if patterns and not any(p in item.name for p in patterns):
            continue
        if item.is_file():
            clear_file(item)
            count += 1
        elif item.is_dir():
            # Optional: recurse only if needed (e.g. chroma_db snapshots)
            pass
    print(f"Cleared {count} files in {directory}")

def main():
    print("Atlas AI Agent – Session Cleanup")
    print("═══════════════════════════════════\n")

    # 1. Kill all mlx_lm.server instances
    mlx_patterns = ["mlx_lm.server", "mlx_lm generate"]
    killed = kill_processes_by_name(mlx_patterns)
    print(f"Killed {killed} mlx_lm.server processes\n")

    # 2. Also target proxy / router if still running (python + proxy.py, etc.)
    proxy_patterns = ["python.*proxy.py", "uvicorn"]
    killed_proxy = kill_processes_by_name(proxy_patterns)
    print(f"Killed {killed_proxy} proxy-related processes\n")

    # 3. Remove active model state
    clear_file(STATE_FILE)

    # 4. Clear log files (keep setup/start logs if you want; here we clear dynamic ones)
    log_patterns = [
        "server_",          # model servers
        "proxy.log",
        "router.log",
        "model_manager.log",
        "start.log"         # optional – comment out if you want to preserve
    ]
    clear_directory_contents(LOGS_DIR, log_patterns)

    # 5. Optional: clear chroma_db if you want full reset (usually not needed)
    # print("Skipping chroma_db wipe (persistent memory preserved)")
    # if input("Wipe chroma_db too? [y/N]: ").lower().startswith('y'):
    #     clear_directory_contents(MEMORY_DIR)

    print("\nCleanup complete.")
    print("You can now safely run python start.py again.\n")

if __name__ == "__main__":
    if os.geteuid() == 0:
        print("Warning: Do not run as root/sudo unless necessary.")
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)