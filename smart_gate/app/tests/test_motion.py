import os
import sys
import pytest
import numpy as np
import cv2
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import motion as motion_module
from motion import (
    _load_privacy_zones,
    apply_privacy_mask,
    _build_exclusion_mask,
    validate_motion_outside_zones,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _make_frame(h=480, w=640, color=(200, 200, 200)):
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _reset_motion_ref():
    """Reset the global reference frame between tests."""
    with motion_module._motion_ref_lock:
        motion_module._motion_ref_frame = None


def _set_zones(zones):
    """Directly set the in-memory privacy zones for testing."""
    with motion_module._zones_lock:
        motion_module._privacy_zones = zones


# ─────────────────────────────────────────────
# _load_privacy_zones
# ─────────────────────────────────────────────


class TestLoadPrivacyZones:

    def test_loads_zones_from_options(self):
        zones = [{"label": "Test", "x1": 0, "y1": 0, "x2": 100, "y2": 100}]
        with patch("motion.get_options", return_value={"privacy_zones": zones}):
            _load_privacy_zones()
        with motion_module._zones_lock:
            assert motion_module._privacy_zones == zones

    def test_empty_zones_when_not_in_options(self):
        with patch("motion.get_options", return_value={}):
            _load_privacy_zones()
        with motion_module._zones_lock:
            assert motion_module._privacy_zones == []

    def test_handles_get_options_exception(self):
        with patch("motion.get_options", side_effect=Exception("read error")):
            # should not raise
            _load_privacy_zones()

    def test_multiple_zones(self):
        zones = [
            {"label": "Zone A", "x1": 0, "y1": 0, "x2": 100, "y2": 100},
            {"label": "Zone B", "x1": 200, "y1": 200, "x2": 300, "y2": 300},
        ]
        with patch("motion.get_options", return_value={"privacy_zones": zones}):
            _load_privacy_zones()
        with motion_module._zones_lock:
            assert len(motion_module._privacy_zones) == 2


# ─────────────────────────────────────────────
# apply_privacy_mask
# ─────────────────────────────────────────────


class TestApplyPrivacyMask:

    def test_blacks_out_zone(self):
        _set_zones([{"label": "Test", "x1": 0, "y1": 0, "x2": 100, "y2": 100}])
        frame = _make_frame()
        result = apply_privacy_mask(frame)
        # Top-left area should be black
        assert np.all(result[0:100, 0:100] == 0)
        # Area outside zone should be untouched
        assert not np.all(result[200:300, 200:300] == 0)

    def test_no_zones_returns_frame_unchanged(self):
        _set_zones([])
        frame = _make_frame()
        original = frame.copy()
        result = apply_privacy_mask(frame)
        np.testing.assert_array_equal(result, original)

    def test_multiple_zones(self):
        _set_zones(
            [
                {"label": "A", "x1": 0, "y1": 0, "x2": 50, "y2": 50},
                {"label": "B", "x1": 200, "y1": 200, "x2": 250, "y2": 250},
            ]
        )
        frame = _make_frame()
        result = apply_privacy_mask(frame)
        assert np.all(result[0:50, 0:50] == 0)
        assert np.all(result[200:250, 200:250] == 0)

    def test_invalid_zone_entry_skipped(self):
        _set_zones([{"label": "Bad"}])  # missing x1/y1/x2/y2
        frame = _make_frame()
        original = frame.copy()
        # should not raise, frame should be unchanged
        result = apply_privacy_mask(frame)
        np.testing.assert_array_equal(result, original)


# ─────────────────────────────────────────────
# _build_exclusion_mask
# ─────────────────────────────────────────────


class TestBuildExclusionMask:

    def test_full_mask_when_no_zones(self):
        _set_zones([])
        mask = _build_exclusion_mask((480, 640, 3))
        assert mask.shape == (480, 640)
        assert np.all(mask == 255)

    def test_zone_is_zeroed(self):
        _set_zones([{"label": "Test", "x1": 10, "y1": 10, "x2": 100, "y2": 100}])
        mask = _build_exclusion_mask((480, 640, 3))
        assert np.all(mask[10:100, 10:100] == 0)
        assert np.all(mask[200:300, 200:300] == 255)

    def test_full_frame_masked(self):
        _set_zones([{"label": "All", "x1": 0, "y1": 0, "x2": 640, "y2": 480}])
        mask = _build_exclusion_mask((480, 640, 3))
        assert np.all(mask == 0)


# ─────────────────────────────────────────────
# validate_motion_outside_zones
# ─────────────────────────────────────────────


class TestValidateMotionOutsideZones:

    def setup_method(self):
        _reset_motion_ref()
        _set_zones([])

    def test_none_frame_returns_true(self):
        assert validate_motion_outside_zones(None) is True

    def test_first_call_stores_ref_and_returns_true(self):
        frame = _make_frame()
        assert validate_motion_outside_zones(frame) is True
        with motion_module._motion_ref_lock:
            assert motion_module._motion_ref_frame is not None

    def test_identical_frames_return_false(self):
        frame = _make_frame()
        validate_motion_outside_zones(frame)  # store ref
        # same frame again — no motion
        assert validate_motion_outside_zones(frame.copy()) is False

    def test_different_frames_return_true(self):
        ref_frame = _make_frame(color=(100, 100, 100))
        validate_motion_outside_zones(ref_frame)  # store ref
        # significantly different frame
        new_frame = _make_frame(color=(0, 0, 0))
        assert validate_motion_outside_zones(new_frame) is True

    def test_motion_inside_zone_ignored(self):
        # Mask the entire frame — all motion should be ignored
        _set_zones([{"label": "All", "x1": 0, "y1": 0, "x2": 640, "y2": 480}])
        ref_frame = _make_frame(color=(100, 100, 100))
        validate_motion_outside_zones(ref_frame)
        # Completely different frame, but all inside masked zone
        new_frame = _make_frame(color=(0, 0, 0))
        assert validate_motion_outside_zones(new_frame) is False

    def test_shape_change_resets_ref(self):
        frame_a = _make_frame(h=480, w=640)
        validate_motion_outside_zones(frame_a)
        frame_b = _make_frame(h=720, w=1280)
        # shape change → resets ref, returns True
        assert validate_motion_outside_zones(frame_b) is True
