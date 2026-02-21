import os

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")
HASS_URL = os.environ.get("HASS_URL", "http://supervisor/core/api")
LOG_PATH = "/config/www/smart_gate/access_log.json"
MAX_ENTRIES = 200
HEADERS = {
    "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
    "Content-Type": "application/json"
}
