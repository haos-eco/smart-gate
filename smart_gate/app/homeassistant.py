import os
import requests

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")
HASS_URL = "http://supervisor/core/api"
HEADERS = {
    "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
    "Content-Type": "application/json"
}

def get_state(entity_id: str) -> str:
    """Get state of a Home Assistant entity"""
    response = requests.get(
        f"{HASS_URL}/states/{entity_id}",
        headers=HEADERS,
        timeout=10
    )
    response.raise_for_status()
    return response.json().get("state", "")

def call_service(service: str, data: dict):
    """Call a Home Assistant service"""
    domain, svc = service.split(".", 1)
    response = requests.post(
        f"{HASS_URL}/services/{domain}/{svc}",
        headers=HEADERS,
        json=data,
        timeout=15
    )
    response.raise_for_status()
    return response.json()

def camera_snapshot(camera_entity: str, path: str):
    """Capture camera snapshot"""
    call_service("camera.snapshot", {
        "entity_id": camera_entity,
        "filename": path
    })

def switch_on(entity_id: str):
    """Turn on a switch"""
    call_service("switch.turn_on", {"entity_id": entity_id})

def send_visitor_notification(notify_devices: list, snapshot_path: str, camera_entity: str, notification_sound: str):
    """
    Send actionable notification to one or more devices.
    notify_devices: list of service strings e.g. ["notify.mobile_app_iphone_di_andrea"]
    snapshot_path: absolute path on HA filesystem e.g. /config/www/smart_gate/snapshot/latest.jpg
    The image is served via /local/... mapped from /config/www/...
    """
    image_url = snapshot_path.replace("/config/www/", "/local/")

    payload = {
        "title": "🚗 Smart Gate",
        "message": "C'è qualcuno all'ingresso",
        "data": {
            "image": image_url,
            "actions": [
                {
                    "action": "SMART_GATE_OPEN",
                    "title": "🔓 Apri Cancello",
                    "destructive": False,
                }
            ],
            "entity_id": camera_entity,
            "url": f"entityId:{camera_entity}",
            "push": {
                "sound": notification_sound
            }
        }
    }

    for service in notify_devices:
        try:
            call_service(service, payload)
            print(f"🔔 Notification sent to {service}")
        except Exception as e:
            print(f"⚠️  Failed to notify {service}: {e}")

def poll_notification_action(action_id: str = "SMART_GATE_OPEN", timeout: int = 120) -> bool:
    """
    Poll HA event bus for a mobile_app_notification_action event matching action_id.
    Returns True if action was fired within timeout seconds, False otherwise.
    Uses /api/events endpoint with long-polling via repeated short requests.
    """
    import time

    # We use the /api/states trick: fire a persistent_notification and watch for
    # mobile_app_notification_action via the logbook or event stream.
    # Simplest reliable approach: poll a dedicated input_boolean or use the event stream.
    #
    # Here we use the HA event stream API (text/event-stream) with a short read timeout.
    deadline = time.time() + timeout
    try:
        with requests.get(
                f"{HASS_URL}/stream",
                headers={**HEADERS, "Accept": "text/event-stream"},
                stream=True,
                timeout=timeout
        ) as resp:
            for line in resp.iter_lines():
                if time.time() > deadline:
                    break
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="ignore")
                if "mobile_app_notification_action" in decoded and action_id in decoded:
                    return True
    except Exception:
        pass
    return False
