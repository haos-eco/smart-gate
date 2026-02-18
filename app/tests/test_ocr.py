"""Tests for OCR module"""
import os
import sys
import pytest
from unittest.mock import Mock  # ← Aggiungi questo import

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr import (
    extract_plate_pattern,
    fix_common_ocr_errors,
    ocr_plate
)

class TestExtractPlatePattern:
    """Tests for plate pattern extraction"""

    def test_valid_7_char_plate(self):
        """Test extraction of valid 7-character plate"""
        assert extract_plate_pattern('GR571XC') == 'GR571XC'

    def test_8_char_with_prefix(self):
        """Test extraction from 8 chars with prefix"""
        assert extract_plate_pattern('IGR571XC') == 'GR571XC'

    def test_8_char_with_suffix(self):
        """Test extraction from 8 chars with suffix"""
        assert extract_plate_pattern('GR571XCI') == 'GR571XC'

    def test_pattern_in_middle(self):
        """Test extraction of pattern from middle of string"""
        assert extract_plate_pattern('XXGR571XCYY') == 'GR571XC'

    def test_invalid_pattern(self):
        """Test with invalid pattern"""
        result = extract_plate_pattern('INVALID')
        assert result == 'INVALID'

class TestFixCommonOCRErrors:
    """Tests for OCR error correction"""

    def test_fix_digit_in_letters(self):
        """Test fixing digits in letter positions"""
        assert fix_common_ocr_errors('0R571XC') == 'OR571XC'
        assert fix_common_ocr_errors('GR5711C') == 'GR571IC'

    def test_fix_letter_in_numbers(self):
        """Test fixing letters in number positions"""
        assert fix_common_ocr_errors('GRO71XC') == 'GR071XC'
        assert fix_common_ocr_errors('GR5I1XC') == 'GR511XC'
        assert fix_common_ocr_errors('GR5Z1XC') == 'GR541XC'

    def test_no_fix_needed(self):
        """Test with correct plate"""
        assert fix_common_ocr_errors('GR571XC') == 'GR571XC'

    def test_wrong_length(self):
        """Test with wrong length (should return as-is)"""
        assert fix_common_ocr_errors('ABC123') == 'ABC123'

def test_ocr_plate_integration(sample_plate_image, mock_easyocr_reader):
    """Integration test for OCR pipeline"""
    result = ocr_plate(mock_easyocr_reader, sample_plate_image, debug=False)

    assert result == 'GR571XC'
    mock_easyocr_reader.readtext.assert_called_once()

def test_ocr_plate_no_results(sample_plate_image):
    """Test OCR with no results"""
    mock_reader = Mock()  # ← Cambiato da pytest.Mock()
    mock_reader.readtext.return_value = []

    result = ocr_plate(mock_reader, sample_plate_image, debug=False)

    assert result == ''

def test_ocr_plate_low_confidence(sample_plate_image):
    """Test OCR with low confidence results"""
    mock_reader = Mock()  # ← Cambiato da pytest.Mock()
    mock_reader.readtext.return_value = [
        ([(0, 0), (100, 0), (100, 50), (0, 50)], 'GR571XC', 0.3)  # Below 0.5
    ]

    result = ocr_plate(mock_reader, sample_plate_image, debug=False)

    assert result == ''

def test_ocr_plate_multiple_results(sample_plate_image):
    """Test OCR with multiple results (picks longest)"""
    mock_reader = Mock()
    mock_reader.readtext.return_value = [
        ([(0, 0), (50, 0), (50, 30), (0, 30)], 'GR', 0.9),
        ([(60, 0), (150, 0), (150, 30), (60, 30)], 'GR571XC', 0.95),
        ([(160, 0), (200, 0), (200, 30), (160, 30)], '571', 0.85)
    ]

    result = ocr_plate(mock_reader, sample_plate_image, debug=False)

    # Should pick the longest valid result
    assert result == 'GR571XC'

def test_ocr_plate_with_extraction(sample_plate_image):
    """Test OCR with pattern extraction (8 chars -> 7 chars)"""
    mock_reader = Mock()
    mock_reader.readtext.return_value = [
        ([(0, 0), (100, 0), (100, 50), (0, 50)], 'IGR571XC', 0.9)  # 8 chars
    ]

    result = ocr_plate(mock_reader, sample_plate_image, debug=False)

    # Should extract 7-char pattern
    assert result == 'GR571XC'

def test_ocr_plate_with_correction(sample_plate_image):
    """Test OCR with error correction"""
    mock_reader = Mock()
    mock_reader.readtext.return_value = [
        ([(0, 0), (100, 0), (100, 50), (0, 50)], 'GR5Z1XC', 0.9)  # Z should be 4
    ]

    result = ocr_plate(mock_reader, sample_plate_image, debug=False)

    # Should fix Z->4 in number position
    assert result == 'GR541XC'
