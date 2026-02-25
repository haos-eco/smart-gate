import os
import sys
import pytest

from unittest.mock import patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from notifications import send_visitor_notification

if os.path.exists("smart_gate"):
    SNAPSHOT_PATH = "smart_gate/app/tests/fixtures/roi/latest.jpg"
else:
    SNAPSHOT_PATH = "tests/fixtures/roi/latest.jpg"

NOTIFICATION_SOUND = "default"


@pytest.fixture
def snapshot():
    assert os.path.exists(SNAPSHOT_PATH), (
        f"Car fixture not found: {SNAPSHOT_PATH}\n"
        "Run test_roi_snapshot.py first to generate it."
    )
    return SNAPSHOT_PATH


def _captured_payload(mock_call, call_index=0):
    """
    Extract the payload dict from a mocked call_service invocation.
    call_service(service, payload) → args = (service, payload)
    """
    return mock_call.call_args_list[call_index].args[1]


def test_notification_sent_to_all_devices(snapshot):
    """send_visitor_notification calls call_service once per device."""
    devices = [
        "notify.mobile_app_iphone_di_andrea",
        "notify.mobile_app_iphone_di_roberto",
    ]

    with patch("notifications.call_service") as mock_call:
        send_visitor_notification(devices, snapshot, NOTIFICATION_SOUND)

    assert mock_call.call_count == len(devices)
    called_services = [c.args[0] for c in mock_call.call_args_list]
    assert called_services == devices


def test_notification_payload_structure(snapshot):
    """Payload contains required fields with correct default title and message."""
    with patch("notifications.call_service") as mock_call:
        send_visitor_notification(
            ["notify.mobile_app_iphone_di_andrea"],
            snapshot,
            NOTIFICATION_SOUND,
        )

    payload = _captured_payload(mock_call)
    data = payload["data"]

    assert payload["title"] == "🚗 Smart Gate"
    assert payload["message"] == "C'è qualcuno all'ingresso"
    assert "image" in data
    assert "actions" in data
    assert "url" in data


def test_notification_custom_title_and_message(snapshot):
    """Custom title and message override defaults (used for failure notifications)."""
    custom_title = "⚠️ SmartGate: targa sconosciuta (AB123CD)"
    custom_message = "Veicolo non in whitelist."

    with patch("notifications.call_service") as mock_call:
        send_visitor_notification(
            ["notify.mobile_app_iphone_di_andrea"],
            snapshot,
            NOTIFICATION_SOUND,
            title=custom_title,
            message=custom_message,
        )

    payload = _captured_payload(mock_call)
    assert payload["title"] == custom_title
    assert payload["message"] == custom_message


def test_notification_image_url_mapping(snapshot):
    """/config/www/ path is correctly mapped to /local/ for HA serving."""
    with patch("notifications.call_service") as mock_call:
        send_visitor_notification(
            ["notify.mobile_app_iphone_di_andrea"],
            "/config/www/smart_gate/snapshot/latest.jpg",
            NOTIFICATION_SOUND,
        )

    payload = _captured_payload(mock_call)
    assert payload["data"]["image"] == "/local/smart_gate/snapshot/latest.jpg"


def test_notification_action_open_gate(snapshot):
    """Payload includes SMART_GATE_OPEN action with correct label."""
    with patch("notifications.call_service") as mock_call:
        send_visitor_notification(
            ["notify.mobile_app_iphone_di_andrea"],
            snapshot,
            NOTIFICATION_SOUND,
        )

    actions = _captured_payload(mock_call)["data"]["actions"]
    assert len(actions) == 1
    assert actions[0]["action"] == "SMART_GATE_OPEN"
    assert actions[0]["destructive"] is False


def test_notification_tap_url_points_to_lovelace(snapshot):
    """Tap URL must point to the smart-gate lovelace view."""
    with patch("notifications.call_service") as mock_call:
        send_visitor_notification(
            ["notify.mobile_app_iphone_di_andrea"],
            snapshot,
            NOTIFICATION_SOUND,
        )

    url = _captured_payload(mock_call)["data"]["url"]
    assert url == "/lovelace/smart-gate"


def test_notification_continues_on_single_device_failure(snapshot):
    """If one device fails, notification is still sent to remaining devices."""
    devices = [
        "notify.mobile_app_iphone_di_andrea",
        "notify.mobile_app_iphone_di_roberto",
    ]

    def fail_first(service, payload):
        if service == devices[0]:
            raise Exception("Device unreachable")

    with patch("notifications.call_service", side_effect=fail_first) as mock_call:
        # Should not raise
        send_visitor_notification(devices, snapshot, NOTIFICATION_SOUND)

    assert mock_call.call_count == len(devices)
