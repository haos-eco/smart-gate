import re
import cv2
import numpy as np

def extract_plate_pattern(text):
    """Extract AA123AA pattern from longer strings"""
    if len(text) == 7:
        if text[0:2].isalpha() and text[2:5].isdigit() and text[5:7].isalpha():
            return text

    # Search for exact pattern
    match = re.search(r'[A-Z]{2}\d{3}[A-Z]{2}', text)
    if match:
        return match.group()

    # Try removing first/last character if 8 chars
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
    """Perform OCR on plate crop"""
    from image_processing import crop_white_area

    # Crop white area (remove EU strip)
    img_bgr = crop_white_area(img_bgr)
    h, w = img_bgr.shape[:2]

    # Aggressive upscaling
    target_h = 400
    if h < target_h:
        scale = target_h / h
        img_bgr = cv2.resize(
            img_bgr, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_LANCZOS4
        )
        if debug:
            print(f"Upscaled to: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # Sharpening
    kernel_sharpen = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    img_bgr = cv2.filter2D(img_bgr, -1, kernel_sharpen)

    # EasyOCR
    results = reader.readtext(
        img_bgr,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    if debug:
        print(f"EasyOCR found {len(results)} text regions:")
        for bbox, text, confidence in results:
            text_clean = re.sub(r"[^A-Z0-9]", "", text.upper())
            print(f"  '{text}' -> '{text_clean}' (confidence: {confidence:.3f})")

    # Filter valid results
    valid_results = [
        (re.sub(r"[^A-Z0-9]", "", text.upper()), conf)
        for _, text, conf in results if conf > 0.5
    ]

    if not valid_results:
        return ""

    # Get best result
    valid_results.sort(key=lambda x: len(x[0]), reverse=True)
    best = valid_results[0][0]
    best_extracted = extract_plate_pattern(best)
    best_fixed = fix_common_ocr_errors(best_extracted)

    if debug:
        print(f"Best OCR: '{best}' -> extracted: '{best_extracted}' -> fixed: '{best_fixed}'")

    return best_fixed
