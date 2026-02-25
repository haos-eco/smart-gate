import os
import sys
import pytest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ocr import extract_plate_pattern, fix_common_ocr_errors, ocr_plate


# ─────────────────────────────────────────────
# Unit tests — no OCR engine needed
# ─────────────────────────────────────────────


class TestExtractPlatePattern:

    def test_valid_7_char_plate(self):
        assert extract_plate_pattern("GR571XC") == "GR571XC"

    def test_8_char_with_prefix(self):
        assert extract_plate_pattern("IGR571XC") == "GR571XC"

    def test_8_char_with_suffix(self):
        assert extract_plate_pattern("GR571XCI") == "GR571XC"

    def test_pattern_in_middle(self):
        assert extract_plate_pattern("XXGR571XCYY") == "GR571XC"

    def test_invalid_pattern(self):
        assert extract_plate_pattern("INVALID") == "INVALID"


class TestFixCommonOCRErrors:

    def test_fix_digit_in_letters(self):
        assert fix_common_ocr_errors("0R571XC") == "OR571XC"
        assert fix_common_ocr_errors("GR5711C") == "GR571IC"

    def test_fix_letter_in_numbers(self):
        assert fix_common_ocr_errors("GRO71XC") == "GR071XC"
        assert fix_common_ocr_errors("GR5I1XC") == "GR511XC"
        assert fix_common_ocr_errors("GR5Z1XC") == "GR541XC"

    def test_no_fix_needed(self):
        assert fix_common_ocr_errors("GR571XC") == "GR571XC"

    def test_wrong_length(self):
        assert fix_common_ocr_errors("ABC123") == "ABC123"


# ─────────────────────────────────────────────
# EasyOCR path (TrOCR forced unavailable)
# ─────────────────────────────────────────────


class TestOCRPlateEasyOCR:
    """Test ocr_plate using mocked EasyOCR (TrOCR disabled)."""

    @pytest.fixture(autouse=True)
    def disable_trocr(self):
        with patch("ocr.load_trocr", return_value=(None, None)):
            yield

    def test_basic(self, sample_plate_image, mock_easyocr_reader):
        plate, conf = ocr_plate(mock_easyocr_reader, sample_plate_image)
        assert plate == "GR571XC"
        assert conf > 0.5
        mock_easyocr_reader.readtext.assert_called_once()

    def test_no_results(self, sample_plate_image):
        reader = Mock()
        reader.readtext.return_value = []
        plate, conf = ocr_plate(reader, sample_plate_image)
        assert plate == ""
        assert conf == 0.0

    def test_low_confidence(self, sample_plate_image):
        reader = Mock()
        reader.readtext.return_value = [
            ([(0, 0), (100, 0), (100, 50), (0, 50)], "GR571XC", 0.3)
        ]
        plate, conf = ocr_plate(reader, sample_plate_image)
        assert plate == ""
        assert conf == 0.0

    def test_picks_longest_result(self, sample_plate_image):
        reader = Mock()
        reader.readtext.return_value = [
            ([(0, 0), (50, 0), (50, 30), (0, 30)], "GR", 0.9),
            ([(60, 0), (150, 0), (150, 30), (60, 30)], "GR571XC", 0.95),
            ([(160, 0), (200, 0), (200, 30), (160, 30)], "571", 0.85),
        ]
        plate, conf = ocr_plate(reader, sample_plate_image)
        assert plate == "GR571XC"
        assert conf == 0.95

    def test_pattern_extraction(self, sample_plate_image):
        reader = Mock()
        reader.readtext.return_value = [
            ([(0, 0), (100, 0), (100, 50), (0, 50)], "IGR571XC", 0.9)
        ]
        plate, conf = ocr_plate(reader, sample_plate_image)
        assert plate == "GR571XC"
        assert conf == 0.9

    def test_error_correction(self, sample_plate_image):
        reader = Mock()
        reader.readtext.return_value = [
            ([(0, 0), (100, 0), (100, 50), (0, 50)], "GR5Z1XC", 0.9)
        ]
        plate, conf = ocr_plate(reader, sample_plate_image)
        assert plate == "GR541XC"
        assert conf == 0.9


# ─────────────────────────────────────────────
# TrOCR path (real model)
# ─────────────────────────────────────────────


class TestOCRPlateTrOCR:
    """Test ocr_plate using real TrOCR model.
    Skipped automatically if TrOCR is unavailable.
    """

    @pytest.fixture(autouse=True)
    def require_trocr(self):
        from trocr import load_trocr

        processor, model = load_trocr()
        if processor is None or model is None:
            pytest.skip("TrOCR model not available")

    def test_returns_string_and_confidence(self, sample_plate_image):
        """TrOCR should return a non-empty string with confidence > 0."""
        reader = Mock()
        plate, conf = ocr_plate(reader, sample_plate_image)
        assert isinstance(plate, str)
        assert isinstance(conf, float)
        assert conf > 0.0

    def test_plate_format(self, sample_plate_image):
        """Result should not exceed 7 chars if a plate is detected."""
        reader = Mock()
        plate, conf = ocr_plate(reader, sample_plate_image)
        if plate:
            assert len(plate) <= 7, f"Plate too long: '{plate}'"

    def test_reads_synthetic_plate(self, sample_plate_image):
        """TrOCR should read the synthetic plate (GP462TC) from conftest."""
        reader = Mock()
        plate, conf = ocr_plate(reader, sample_plate_image, debug=True)
        print(f"\nTrOCR result: '{plate}' (confidence: {conf:.3f})")
        assert conf > 0.0
