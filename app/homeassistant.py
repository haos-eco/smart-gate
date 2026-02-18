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
