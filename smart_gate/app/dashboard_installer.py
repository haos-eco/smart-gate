import os
import shutil
import yaml
import requests
from constants import (
    DASHBOARD_SRC,
    DASHBOARD_DST,
    DASHBOARD_URL_PATH,
    LOVELACE_DASHBOARDS_FILE,
    HASS_URL,
    HEADERS,
)


def install_dashboard() -> None:
    """
    1. Copy bundled HTML files from DASHBOARD_SRC to DASHBOARD_DST
    2. Register Smart Gate dashboard in /config/dashboards.yaml
    3. Reload Lovelace via HA API so the sidebar link appears immediately
    """
    _copy_files()
    _register_dashboard_yaml()
    _reload_lovelace()


# ── Step 1: copy HTML files ───────────────────────────────────────────────────


def _copy_files() -> None:
    if not os.path.isdir(DASHBOARD_SRC):
        print(
            f"[SmartGate] Dashboard source not found at {DASHBOARD_SRC} — skipping install"
        )
        return

    if os.path.exists(DASHBOARD_DST):
        shutil.rmtree(DASHBOARD_DST)
        print(f"[SmartGate] Removed existing dashboard at {DASHBOARD_DST}")

    shutil.copytree(DASHBOARD_SRC, DASHBOARD_DST)
    print(f"[SmartGate] Dashboard files installed: {DASHBOARD_SRC} → {DASHBOARD_DST}")


# ── Step 2: register in dashboards.yaml ──────────────────────────────────────


def _register_dashboard_yaml() -> None:
    """
    Add Smart Gate entry to /config/dashboards.yaml.
    HA reads this file at startup to populate the sidebar.
    If the entry already exists it is left unchanged.
    """
    entry = {
        "smart-gate": {
            "mode": "yaml",
            "filename": "www/smart_gate/dashboard/dashboard.yaml",
            "title": "Smart Gate",
            "icon": "mdi:gate",
            "show_in_sidebar": True,
            "require_admin": False,
        }
    }

    # Load existing dashboards.yaml or start fresh
    existing = {}
    if os.path.exists(LOVELACE_DASHBOARDS_FILE):
        try:
            with open(LOVELACE_DASHBOARDS_FILE) as f:
                existing = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[SmartGate] Could not read {LOVELACE_DASHBOARDS_FILE}: {e}")

    if DASHBOARD_URL_PATH in existing:
        print("[SmartGate] Dashboard entry already present in dashboards.yaml")
        return

    existing.update(entry)

    try:
        with open(LOVELACE_DASHBOARDS_FILE, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
        print(f"[SmartGate] Dashboard registered in {LOVELACE_DASHBOARDS_FILE}")
    except Exception as e:
        print(f"[SmartGate] Could not write {LOVELACE_DASHBOARDS_FILE}: {e}")


# ── Step 3: reload Lovelace ───────────────────────────────────────────────────


def _reload_lovelace() -> None:
    """
    Tell HA to reload Lovelace resources so the new dashboard
    appears in the sidebar without a manual restart.
    """
    try:
        resp = requests.post(
            f"{HASS_URL}/services/lovelace/reload_resources",
            headers=HEADERS,
            json={},
            timeout=10,
        )
        resp.raise_for_status()
        print("[SmartGate] Lovelace reloaded")
    except Exception as e:
        print(f"[SmartGate] Lovelace reload failed (non-critical): {e}")
        print("[SmartGate] Restart Home Assistant once to apply the new dashboard")
