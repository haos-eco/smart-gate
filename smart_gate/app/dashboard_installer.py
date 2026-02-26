import os
import shutil
import yaml
import requests
from constants import DASHBOARD_SRC, DASHBOARD_DST, HASS_URL, HEADERS

# Lovelace dashboard url_path — must match the path in dashboard.yaml views
DASHBOARD_URL_PATH = "smart-gate"


def install_dashboard() -> None:
    """
    1. Copy bundled HTML files from DASHBOARD_SRC to DASHBOARD_DST
    2. Create the Lovelace dashboard via HA API (if not already present)
    3. Push the dashboard config (views) from dashboard.yaml
    """
    _copy_files()
    _register_lovelace_dashboard()


# ── Step 1: copy files ────────────────────────────────────────────────────────


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


# ── Step 2: register Lovelace dashboard + sidebar ────────────────────────────


def _register_lovelace_dashboard() -> None:
    """
    Create the Smart Gate Lovelace dashboard with sidebar link via HA API.
    If the dashboard already exists, only pushes updated config.
    """
    try:
        if not _dashboard_exists():
            _create_dashboard()
        _push_dashboard_config()
    except Exception as e:
        print(f"[SmartGate] Dashboard registration error: {e}")
        print(
            "[SmartGate] Dashboard files are installed — add the panel manually in HA if needed"
        )


def _dashboard_exists() -> bool:
    resp = requests.get(
        f"{HASS_URL}/lovelace/dashboards",
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    dashboards = resp.json()
    return any(d.get("url_path") == DASHBOARD_URL_PATH for d in dashboards)


def _create_dashboard() -> None:
    payload = {
        "url_path": DASHBOARD_URL_PATH,
        "title": "Smart Gate",
        "icon": "mdi:gate",
        "show_in_sidebar": True,
        "require_admin": False,
    }
    resp = requests.post(
        f"{HASS_URL}/lovelace/dashboards",
        headers=HEADERS,
        json=payload,
        timeout=10,
    )
    resp.raise_for_status()
    print(f"[SmartGate] Lovelace dashboard created (url_path: {DASHBOARD_URL_PATH})")


def _push_dashboard_config() -> None:
    """Read dashboard.yaml from the installed files and push it as the dashboard config."""
    config_path = os.path.join(DASHBOARD_DST, "dashboard.yaml")

    if not os.path.exists(config_path):
        print(
            f"[SmartGate] dashboard.yaml not found at {config_path} — skipping config push"
        )
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    resp = requests.post(
        f"{HASS_URL}/lovelace/dashboards/{DASHBOARD_URL_PATH}/config",
        headers=HEADERS,
        json=config,
        timeout=10,
    )
    resp.raise_for_status()
    print(f"[SmartGate] Lovelace dashboard config pushed")
