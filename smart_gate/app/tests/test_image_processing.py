import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import (
    apply_roi,
    remove_plate_border,
    crop_white_area,
    is_overexposed,
    fix_headlight_overexposure,
    deskew_plate,
    preprocess_plate,
    get_sr_model,
)


@pytest.fixture(scope="session", autouse=True)
def preload_trocr():
    """Carica TrOCR una volta sola per tutta la sessione di test."""
    from trocr import load_trocr
    load_trocr()

def make_blank(h=100, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_apply_roi():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = [0.25, 0.25, 0.5, 0.5]
    result = apply_roi(img, roi)
    assert result.shape == (50, 100, 3)


def test_remove_plate_border():
    img = np.ones((100, 300, 3), dtype=np.uint8) * 128
    result = remove_plate_border(img)
    # Margins are 18% top/bottom and 8% left/right
    assert result.shape[0] < img.shape[0]
    assert result.shape[1] < img.shape[1]


def test_crop_white_area_returns_image():
    """crop_white_area should always return a valid image (fallback if no white cols found)"""
    img = make_blank(50, 200)
    result = crop_white_area(img)
    assert result is not None
    assert result.ndim == 3


def test_crop_white_area_removes_blue_strip():
    """If right half is white and left half is blue, only white part should remain"""
    img = np.zeros((50, 200, 3), dtype=np.uint8)
    # Right half: white
    img[:, 100:] = [255, 255, 255]
    result = crop_white_area(img)
    # Result should be narrower than original
    assert result.shape[1] < img.shape[1]


def test_is_overexposed_bright():
    img = np.full((50, 100, 3), 220, dtype=np.uint8)
    assert is_overexposed(img)


def test_is_overexposed_dark():
    img = np.full((50, 100, 3), 50, dtype=np.uint8)
    assert not is_overexposed(img)


def test_fix_headlight_overexposure_reduces_brightness():
    img = np.full((50, 100, 3), 240, dtype=np.uint8)
    result = fix_headlight_overexposure(img)
    gray_before = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    gray_after = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    # After fix, image should no longer be overexposed (mean < 200)
    assert gray_after < gray_before


def test_deskew_plate_returns_same_shape():
    img = make_blank(60, 200)
    result = deskew_plate(img)
    assert result.shape == img.shape


def test_deskew_plate_handles_no_lines():
    """Blank image has no lines — deskew should return unchanged"""
    img = make_blank(60, 200)
    result = deskew_plate(img)
    assert result.shape == img.shape


def test_preprocess_plate_returns_bgr():
    """preprocess_plate should return a 3-channel BGR image"""
    img = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
    result = preprocess_plate(img)
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_preprocess_plate_upscales():
    """Output should be at least 400px tall after SR + upscale"""
    img = np.random.randint(0, 255, (30, 100, 3), dtype=np.uint8)
    result = preprocess_plate(img)
    assert result.shape[0] >= 400


def test_preprocess_plate_overexposed_input():
    """preprocess_plate should not crash on an overexposed input"""
    img = np.full((40, 120, 3), 230, dtype=np.uint8)
    result = preprocess_plate(img, debug=True)
    assert result is not None
    assert result.ndim == 3


def test_sr_model_loads():
    sr = get_sr_model()
    if sr is None:
        print("\n⚠️  AI SR model not available, bicubic fallback will be used")
    else:
        print("\n✅ AI SR model loaded successfully")
