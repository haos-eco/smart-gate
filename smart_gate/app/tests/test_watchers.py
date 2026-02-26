import os
import sys
import time
import json
import pytest
import threading
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import motion as motion_module


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _get_zones():
    with motion_module._zones_lock:
        return list(motion_module._privacy_zones)


def _set_zones(zones):
    with motion_module._zones_lock:
        motion_module._privacy_zones = zones


# ─────────────────────────────────────────────
# start_config_watcher
# ─────────────────────────────────────────────


class TestConfigWatcher:

    def test_loads_zones_at_startup(self, tmp_path, monkeypatch):
        """Watcher calls _load_privacy_zones immediately on start."""
        zones = [{"label": "A", "x1": 0, "y1": 0, "x2": 100, "y2": 100}]
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps({"privacy_zones": zones}))

        monkeypatch.setattr("watchers.OPTIONS_PATH", str(options_file))
        monkeypatch.setattr(
            "motion.get_options",
            lambda: json.loads(options_file.read_text()),
        )

        from watchers import start_config_watcher

        start_config_watcher(interval=1)

        assert _get_zones() == zones

    def test_reloads_on_file_change(self, tmp_path, monkeypatch):
        """Watcher detects mtime change and reloads zones within poll interval."""
        zones_v1 = [{"label": "A", "x1": 0, "y1": 0, "x2": 100, "y2": 100}]
        zones_v2 = [{"label": "B", "x1": 200, "y1": 200, "x2": 300, "y2": 300}]

        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps({"privacy_zones": zones_v1}))

        monkeypatch.setattr("watchers.OPTIONS_PATH", str(options_file))
        monkeypatch.setattr(
            "motion.get_options",
            lambda: json.loads(options_file.read_text()),
        )

        # Reset watcher state
        import watchers

        monkeypatch.setattr(watchers, "_last_mtime", 0.0)

        from watchers import start_config_watcher

        start_config_watcher(interval=1)

        assert _get_zones() == zones_v1

        # Simulate user saving new config
        time.sleep(0.1)
        options_file.write_text(json.dumps({"privacy_zones": zones_v2}))

        # Wait for watcher to pick up the change (poll interval = 1s)
        deadline = time.time() + 3
        while time.time() < deadline:
            if _get_zones() == zones_v2:
                break
            time.sleep(0.1)

        assert _get_zones() == zones_v2

    def test_no_reload_when_file_unchanged(self, tmp_path, monkeypatch):
        """Watcher does not reload if mtime has not changed."""
        zones = [{"label": "A", "x1": 0, "y1": 0, "x2": 100, "y2": 100}]
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps({"privacy_zones": zones}))

        monkeypatch.setattr("watchers.OPTIONS_PATH", str(options_file))

        load_count = {"n": 0}
        original_load = motion_module._load_privacy_zones

        def counting_load():
            load_count["n"] += 1
            original_load()

        monkeypatch.setattr("watchers._load_privacy_zones", counting_load)
        monkeypatch.setattr(
            "motion.get_options", lambda: json.loads(options_file.read_text())
        )

        import watchers

        monkeypatch.setattr(watchers, "_last_mtime", 0.0)

        from watchers import start_config_watcher

        start_config_watcher(interval=1)

        initial_count = load_count["n"]
        time.sleep(1.5)  # wait one full poll cycle

        # File not changed — count should not have increased
        assert load_count["n"] == initial_count

    def test_handles_missing_options_file(self, tmp_path, monkeypatch):
        """Watcher does not crash if options.json does not exist."""
        monkeypatch.setattr("watchers.OPTIONS_PATH", str(tmp_path / "missing.json"))

        import watchers

        monkeypatch.setattr(watchers, "_last_mtime", 0.0)
        monkeypatch.setattr("motion.get_options", lambda: {})

        from watchers import start_config_watcher

        # should not raise
        start_config_watcher(interval=1)
        time.sleep(1.2)
