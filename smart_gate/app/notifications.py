import time
import threading
import requests
from constants import HASS_URL, HEADERS


def send_visitor_notification(notify_devices: list, snapshot_path: str, camera_entity: str, notification_sound: str):
    """
    Send actionable notification to one or more devices.
    notify_devices: list of service strings e.g. ["notify.mobile_app_iphone_di_andrea"]
    snapshot_path: absolute path on HA filesystem e.g. /config/www/smart_gate/snapshot/latest.jpg
    The image is served via /local/... mapped from /config/www/...
    """
    from homeassistant import call_service

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
    Listen to HA event stream for a mobile_app_notification_action matching action_id.
    Returns True if action was fired within timeout seconds, False otherwise.
    """
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


def handle_notification_action(gate_switch: str, debug: bool = False):
    """Background thread: wait for SMART_GATE_OPEN action, open gate if received."""
    from homeassistant import switch_on
    print("🔔 Waiting for notification action (120s timeout)...")
    fired = poll_notification_action(action_id="SMART_GATE_OPEN", timeout=120)
    if fired:
        print("✅ Gate opened via notification action")
        switch_on(gate_switch)
    else:
        if debug:
            print("🔔 Notification action timeout — no response")


def start_notification_listener(gate_switch: str, debug: bool = False) -> threading.Thread:
    """Spawn and return a daemon thread listening for the open gate notification action."""
    t = threading.Thread(
        target=handle_notification_action,
        args=(gate_switch, debug),
        daemon=True
    )
    t.start()
    return t
