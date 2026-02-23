from trocr import load_trocr, trocr_infer
import re

def extract_plate_pattern(text):
    """Extract AA123AA pattern from longer strings"""
    if len(text) == 7:
        if text[0:2].isalpha() and text[2:5].isdigit() and text[5:7].isalpha():
            return text

    match = re.search(r'[A-Z]{2}\d{3}[A-Z]{2}', text)
    if match:
        return match.group()

    if len(text) == 8:
        candidate = text[1:]
        if candidate[0:2].isalpha() and candidate[2:5].isdigit() and candidate[5:7].isalpha():
            return candidate

        candidate = text[:-1]
        if candidate[0:2].isalpha() and candidate[2:5].isdigit() and candidate[5:7].isalpha():
            return candidate

    return text

def fix_common_ocr_errors(plate_text):
    """Fix common OCR errors for Italian plate format AA123AA"""
    letter_fixes = {'0': 'O', '1': 'I', '4': 'A', '8': 'B'}
    number_fixes = {'O': '0', 'I': '1', 'Z': '4', 'S': '5', 'B': '8'}

    if len(plate_text) != 7:
        return plate_text

    result = list(plate_text)

    # Positions 0,1 should be letters
    for i in [0, 1]:
        if result[i].isdigit() and result[i] in letter_fixes:
            result[i] = letter_fixes[result[i]]

    # Positions 2,3,4 should be numbers
    for i in [2, 3, 4]:
        if result[i].isalpha() and result[i] in number_fixes:
            result[i] = number_fixes[result[i]]

    # Positions 5,6 should be letters
    for i in [5, 6]:
        if result[i].isdigit() and result[i] in letter_fixes:
            result[i] = letter_fixes[result[i]]

    return ''.join(result)

def ocr_plate(reader, img_bgr, debug=False):
    """
    Run OCR on a preprocessed plate crop.
    Tries TrOCR first (more accurate on printed text).
    Falls back to EasyOCR if TrOCR is unavailable.

    Args:
        reader: EasyOCR reader instance (used as fallback)
        img_bgr: preprocessed plate crop (BGR)
        debug: enable debug logging

    Returns:
        (plate_text, confidence)
    """
    processor, model = load_trocr()

    if processor is not None and model is not None:
        return _ocr_with_trocr(img_bgr, debug=debug)
    else:
        if debug:
            print("  Using EasyOCR fallback")
        return _ocr_with_easyocr(reader, img_bgr, debug=debug)


def _ocr_with_trocr(img_bgr, debug=False):
    text, confidence = trocr_infer(img_bgr, debug=debug)

    if not text:
        return "", 0.0

    # Normalize: uppercase, remove non-alphanumeric
    text_clean = re.sub(r"[^A-Z0-9]", "", text.upper())

    if debug:
        print(f"  TrOCR cleaned: '{text_clean}' (confidence: {confidence:.3f})")

    text_extracted = extract_plate_pattern(text_clean)
    text_fixed = fix_common_ocr_errors(text_extracted)

    if debug:
        print(f"  TrOCR final: '{text_fixed}' (confidence: {confidence:.3f})")

    return text_fixed, confidence


def _ocr_with_easyocr(reader, img_bgr, debug=False):
    """OCR using EasyOCR (fallback)."""
    results = reader.readtext(
        img_bgr,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    if debug:
        print(f"  EasyOCR found {len(results)} text regions:")
        for bbox, text, confidence in results:
            text_clean = re.sub(r"[^A-Z0-9]", "", text.upper())
            print(f"    '{text}' -> '{text_clean}' (confidence: {confidence:.3f})")

    valid_results = [
        (re.sub(r"[^A-Z0-9]", "", text.upper()), conf)
        for _, text, conf in results if conf > 0.5
    ]

    if not valid_results:
        return "", 0.0

    valid_results.sort(key=lambda x: len(x[0]), reverse=True)
    best, best_conf = valid_results[0]
    best_extracted = extract_plate_pattern(best)
    best_fixed = fix_common_ocr_errors(best_extracted)

    if debug:
        print(f"  EasyOCR final: '{best_fixed}' (confidence: {best_conf:.3f})")

    return best_fixed, best_conf
