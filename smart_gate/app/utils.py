import os
import time
import json
import cv2
import re as _re
from constants import LOG_PATH


def get_options():
    """Load options from Home Assistant supervisor"""
    with open("/data/options.json", "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def is_complete_plate(p: str) -> bool:
    """Returns True if plate matches full Italian format AA123AA"""
    return bool(_re.match(r"^[A-Z]{2}\d{3}[A-Z]{2}$", p))


def load_logs() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_logs(entries: list):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)


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


def capture_frame(camera_entity, snapshot_path):
    from homeassistant import camera_snapshot

    ensure_dir(snapshot_path)
    camera_snapshot(camera_entity, snapshot_path)
    return cv2.imread(snapshot_path)


def recognize_frame(
    img,
    sess,
    inp,
    out,
    reader,
    roi,
    conf,
    debug,
    logs_dir=None,
    label="detection",
    debug_crop_path=None,
    attempt_number=None,
    ocr_engine="trocr",
    trocr_infer_fn=None,
):
    """
    Detect and recognize a license plate in `img`.

    OCR routing:
      - ocr_engine="trocr"   → calls trocr_infer_fn(crop, debug)
      - ocr_engine="easyocr" → calls ocr_plate(reader, crop, debug) via ocr.py
    """
    from image_processing import apply_roi, remove_plate_border, preprocess_plate
    from detection import detect_plates

    if img is None:
        return None, 0.0, 0.0, None, None

    img_roi = apply_roi(img, roi)
    boxes = detect_plates(sess, inp, out, img_roi, conf=conf, debug=debug)

    if not boxes:
        return None, 0.0, 0.0, None, None

    boxes.sort(key=lambda b: b[4], reverse=True)
    x1, y1, x2, y2, score = boxes[0]
    bbox = (int(x1), int(y1), int(x2), int(y2))
    plate_crop = img_roi[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]

    if plate_crop.size == 0:
        return None, 0.0, 0.0, None, None

    # Preprocessing: denoise → CLAHE → sharpen → deskew → SR → binarize
    plate_crop = remove_plate_border(plate_crop)
    plate_crop = preprocess_plate(plate_crop, debug=debug)

    if debug and debug_crop_path:
        try:
            ensure_dir(debug_crop_path)
            if attempt_number is not None:
                base, ext = os.path.splitext(debug_crop_path)
                save_path = f"{base}_attempt{attempt_number}{ext}"
            else:
                save_path = debug_crop_path
            cv2.imwrite(save_path, plate_crop)
            print(f"  Debug crop saved: {save_path}")
        except Exception as e:
            if debug:
                print(f"  ⚠️  Failed to save debug crop: {e}")

    det_snapshot_path = None
    if logs_dir:
        try:
            os.makedirs(logs_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(logs_dir, f"{label}_{ts}.jpg")
            cv2.imwrite(dest, img)
            det_snapshot_path = dest
        except Exception as e:
            print(f"⚠️  Could not save detection snapshot: {e}")

    if ocr_engine == "trocr" and trocr_infer_fn is not None:
        plate, ocr_conf = trocr_infer_fn(plate_crop, debug)
    else:
        from ocr import ocr_plate

        plate, ocr_conf = ocr_plate(reader, plate_crop, debug=debug)

    return plate, score, ocr_conf, det_snapshot_path, bbox


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


def fuzzy_match(plate: str, allowed, max_distance: int = 2):
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


def cleanup_history(history_dir, logs_dir, keep_days=30):
    removed = 0
    for folder in [history_dir, logs_dir]:
        if not os.path.exists(folder):
            continue
        cutoff = time.time() - (keep_days * 86400)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                removed += 1
    if removed:
        print(f"🧹 Cleanup: rimossi {removed} file più vecchi di {keep_days} giorni")
