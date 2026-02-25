import os

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")
HASS_URL = os.environ.get("HASS_URL", "http://supervisor/core/api")
CONFIG_PATH = "/data/config.yaml"
DASHBOARD_SRC = "/app/dashboard"
DASHBOARD_DST = "/config/www/smart_gate/dashboard"
LOG_PATH = "/config/www/smart_gate/access_log.json"
CLEANUP_INTERVAL = 86400  # 1 day
MAX_ENTRIES = 200

HEADERS = {
    "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
    "Content-Type": "application/json",
}

STATUS_COLORS = {
    "opened": (0, 200, 80),
    "rejected": (0, 80, 255),
    "unknown": (0, 165, 255),
    "unreadable": (180, 180, 180),
}
