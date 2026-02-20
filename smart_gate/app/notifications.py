import threading

def handle_notification_action(gate_switch: str, debug: bool = False):
    """Background thread: wait for SMART_GATE_OPEN action, open gate if received."""
    from homeassistant import poll_notification_action, switch_on
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
