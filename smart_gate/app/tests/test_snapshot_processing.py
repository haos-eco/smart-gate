import os
import sys
import pytest
import numpy as np
import cv2
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snapshot_processing import annotate_snapshot


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


@pytest.fixture
def sample_snapshot(tmp_path):
    """Create a real JPEG snapshot in a temp directory."""
    frame = np.full((480, 640, 3), (150, 150, 150), dtype=np.uint8)
    path = str(tmp_path / "latest.jpg")
    cv2.imwrite(path, frame)
    return path


# ─────────────────────────────────────────────
# annotate_snapshot
# ─────────────────────────────────────────────


class TestAnnotateSnapshot:

    def test_returns_path_in_annotated_subdir(self, sample_snapshot):
        out = annotate_snapshot(sample_snapshot, "AB123CD", None, 0.8, 0.9, "opened")
        assert "annotated" in out
        parent = os.path.dirname(sample_snapshot)
        assert out == os.path.join(
            parent, "annotated", os.path.basename(sample_snapshot)
        )

    def test_annotated_file_exists(self, sample_snapshot):
        out = annotate_snapshot(sample_snapshot, "AB123CD", None, 0.8, 0.9, "opened")
        assert os.path.exists(out)

    def test_original_snapshot_unchanged(self, sample_snapshot):
        original = cv2.imread(sample_snapshot)
        annotate_snapshot(sample_snapshot, "AB123CD", None, 0.8, 0.9, "opened")
        after = cv2.imread(sample_snapshot)
        np.testing.assert_array_equal(original, after)

    def test_annotated_image_is_valid(self, sample_snapshot):
        out = annotate_snapshot(sample_snapshot, "AB123CD", None, 0.8, 0.9, "opened")
        img = cv2.imread(out)
        assert img is not None
        assert img.shape == (480, 640, 3)

    def test_with_bbox(self, sample_snapshot):
        bbox = (100, 100, 300, 200)
        out = annotate_snapshot(sample_snapshot, "AB123CD", bbox, 0.8, 0.9, "opened")
        assert os.path.exists(out)

    def test_all_statuses(self, sample_snapshot, tmp_path):
        for status in ["opened", "rejected", "unknown", "unreadable"]:
            # Each needs a fresh snapshot to avoid file conflicts
            frame = np.full((480, 640, 3), (150, 150, 150), dtype=np.uint8)
            path = str(tmp_path / f"snap_{status}.jpg")
            cv2.imwrite(path, frame)
            out = annotate_snapshot(path, "AB123CD", None, 0.8, 0.9, status)
            assert os.path.exists(out), f"Missing annotated file for status: {status}"

    def test_nonexistent_snapshot_returns_original_path(self):
        fake_path = "/tmp/does_not_exist_12345.jpg"
        out = annotate_snapshot(fake_path, "AB123CD", None, 0.8, 0.9, "opened")
        assert out == fake_path

    def test_status_bar_drawn(self, sample_snapshot):
        """Annotated image should differ from original (status bar adds dark pixels)."""
        original = cv2.imread(sample_snapshot)
        out = annotate_snapshot(sample_snapshot, "AB123CD", None, 0.8, 0.9, "opened")
        annotated = cv2.imread(out)
        assert not np.array_equal(original, annotated)

    def test_annotated_dir_created_automatically(self, tmp_path):
        """annotated/ subdirectory should be created if it doesn't exist."""
        frame = np.full((480, 640, 3), (100, 100, 100), dtype=np.uint8)
        path = str(tmp_path / "latest.jpg")
        cv2.imwrite(path, frame)
        annotated_dir = tmp_path / "annotated"
        assert not annotated_dir.exists()
        annotate_snapshot(path, "AB123CD", None, 0.8, 0.9, "opened")
        assert annotated_dir.exists()
