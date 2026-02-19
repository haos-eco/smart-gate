import os
import json
import cv2

def get_options():
    """Load options from Home Assistant supervisor"""
    with open("/data/options.json", "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def validate_model(model_path: str) -> bool:
    if not os.path.exists(model_path):
        print(f"❌ ERROR: ONNX model not found at: {model_path}")
        print(f"  Please download the model and place it at: {model_path}")
        return False

    file_size = os.path.getsize(model_path) / (1024 * 1024)
    if file_size < 10:
        print(f"⚠️  Model file too small: {file_size:.1f}MB (expected ~200MB)")
        return False

    print(f"✅ Model found: {model_path} ({file_size:.1f}MB)")
    return True

def capture_and_recognize(camera_entity, snapshot_path, sess, inp, out, reader, roi, conf, debug, debug_crop_path=None, attempt_number=None):
    from homeassistant import camera_snapshot
    from image_processing import apply_roi, is_infrared, remove_plate_border, fix_overexposure
    from detection import detect_plates
    from ocr import ocr_plate

    ensure_dir(snapshot_path)
    camera_snapshot(camera_entity, snapshot_path)

    img = cv2.imread(snapshot_path)
    if img is None:
        return None, 0.0

    img_roi = apply_roi(img, roi)
    boxes = detect_plates(sess, inp, out, img_roi, conf=conf, debug=debug)

    if not boxes:
        return None, 0.0

    boxes.sort(key=lambda b: b[4], reverse=True)
    x1, y1, x2, y2, score = boxes[0]
    plate_crop = img_roi[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

    if plate_crop.size == 0:
        return None, 0.0

    # Preprocessing
    plate_crop = remove_plate_border(plate_crop)

    ir_detected, saturation = is_infrared(plate_crop)
    if ir_detected:
        plate_crop = fix_overexposure(plate_crop)
        if debug:
            print(f"  IR mode (sat={saturation:.1f}), applied exposure fix")

    if debug and debug_crop_path:
        try:
            ensure_dir(debug_crop_path)

            if attempt_number is not None:
                base, ext = os.path.splitext(debug_crop_path)
                save_path = f"{base}_attempt{attempt_number}{ext}"
            else:
                save_path = debug_crop_path

            cv2.imwrite(save_path, plate_crop)
            if debug:
                print(f"  Debug crop saved: {save_path}")
        except Exception as e:
            if debug:
                print(f"  ⚠️  Failed to save debug crop: {e}")

    # OCR with AI super-resolution
    plate = ocr_plate(reader, plate_crop, debug=debug)

    return plate, score

def levenshtein(a: str, b: str) -> int:
    """Calculate edit distance between two strings"""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]

def fuzzy_match(plate: str, allowed: list, max_distance: int = 2):
    """
    Returns (matched_plate, distance) if a close match is found, else (None, -1).
    Distance 0 = exact match.
    """
    best_plate = None
    best_dist = max_distance + 1
    for candidate in allowed:
        d = levenshtein(plate, candidate)
        if d < best_dist:
            best_dist = d
            best_plate = candidate
    if best_plate and best_dist <= max_distance:
        return best_plate, best_dist
    return None, -1
