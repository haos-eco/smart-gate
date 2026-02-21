import os
import sys
import shutil
import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from notifications import send_visitor_notification

NOTIFY_DEVICES = [
    "notify.mobile_app_iphone_di_andrea",
]
CAMERA_ENTITY = "camera.ingresso_high_quality"
NOTIFICATION_SOUND = "default"

if os.path.exists('smart_gate'):
    SNAPSHOT_PATH = "smart_gate/app/tests/fixtures/roi/latest.jpg"
else:
    SNAPSHOT_PATH = "tests/fixtures/roi/latest.jpg"

HA_SNAPSHOT_PATH = "/config/www/smart_gate/snapshot/latest.jpg"

def test_send_real_notification():
    """
    Integration test — sends a real push notification to the configured devices.
    Requires SUPERVISOR_TOKEN env var to be set (via .env or export).

    Run:
        pytest -s smart_gate/app/tests/test_notifications_integration.py
    """
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        pytest.skip("SUPERVISOR_TOKEN not set — skipping integration test")

    assert os.path.exists(SNAPSHOT_PATH), (
        f"Car fixture not found: {SNAPSHOT_PATH}\n"
        "Run test_roi_snapshot.py first to generate it."
    )

    # Try to copy fixture to HA www path so the image renders in the notification.
    # If /config is not writable (local dev), fall back to the local fixture path —
    # notification will be sent but image won't render on phone.
    try:
        os.makedirs(os.path.dirname(HA_SNAPSHOT_PATH), exist_ok=True)
        shutil.copy(SNAPSHOT_PATH, HA_SNAPSHOT_PATH)
        effective_path = HA_SNAPSHOT_PATH
        print(f"\n📸 Fixture copied to {HA_SNAPSHOT_PATH}")
    except OSError:
        effective_path = SNAPSHOT_PATH
        print(f"\n⚠️  Could not write to {HA_SNAPSHOT_PATH} — sending without image")

    print(f"📤 Sending test notification to: {NOTIFY_DEVICES}")
    send_visitor_notification(NOTIFY_DEVICES, effective_path, CAMERA_ENTITY, NOTIFICATION_SOUND)
    print("✅ Notification sent — check your phone")
