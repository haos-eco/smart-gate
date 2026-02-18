import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import (
    apply_roi,
    is_infrared,
    remove_plate_border,
    fix_overexposure,
    crop_white_area
)

def test_apply_roi():
    """Test ROI application"""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = [0.25, 0.25, 0.5, 0.5]
    result = apply_roi(img, roi)
    assert result.shape == (50, 100, 3)

# ... resto dei test
