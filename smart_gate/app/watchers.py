import os
import time
import threading

from constants import OPTIONS_PATH
from motion import _load_privacy_zones

# HA writes options to /data/options.json when the user saves addon config
_last_mtime = 0.0


def start_config_watcher(interval: int = 5) -> None:
    """
    Poll /data/options.json every `interval` seconds.
    Reloads privacy_zones only when the file mtime changes
    (i.e. when the user saves the addon configuration in HA).
    Runs as a daemon thread — no extra dependencies required.
    """
    global _last_mtime
    _load_privacy_zones()  # immediate load at startup

    def _watch():
        global _last_mtime
        while True:
            try:
                mtime = os.path.getmtime(OPTIONS_PATH)
                if mtime != _last_mtime:
                    _last_mtime = mtime
                    _load_privacy_zones()
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[SmartGate] Config watcher error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    print(f"[SmartGate] Config watcher started (polling every {interval}s)")
