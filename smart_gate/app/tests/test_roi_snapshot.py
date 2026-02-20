import os
import sys
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import apply_roi

SNAPSHOT_PATH = "tests/fixtures/roi/latest.jpg"
ROI = [0.25, 0.15, 0.55, 0.50]
OUTPUT_PATH = "tests/fixtures/roi/latest_roi_output.jpg"

def test_roi_crop_produces_output():
    """Crop snapshot using production ROI and save result for visual inspection."""
    assert os.path.exists(SNAPSHOT_PATH), f"Snapshot not found: {SNAPSHOT_PATH}"

    img = cv2.imread(SNAPSHOT_PATH)
    assert img is not None, "Failed to read snapshot"

    original_h, original_w = img.shape[:2]
    print(f"\nOriginal size: {original_w}x{original_h}")

    cropped = apply_roi(img, ROI)
    cropped_h, cropped_w = cropped.shape[:2]
    print(f"Cropped size:  {cropped_w}x{cropped_h}")
    print(f"ROI applied:   x={ROI[0]}, y={ROI[1]}, w={ROI[2]}, h={ROI[3]}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, cropped)
    print(f"Output saved:  {OUTPUT_PATH}")

    assert os.path.exists(OUTPUT_PATH), "Output file was not created"
    assert cropped_w > 0 and cropped_h > 0
