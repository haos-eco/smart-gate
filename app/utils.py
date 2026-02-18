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

def capture_and_recognize(camera_entity, snapshot_path, sess, inp, out, reader, roi, conf, debug):
    """Capture snapshot and recognize plate (single attempt)"""
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

    # OCR with AI super-resolution
    plate = ocr_plate(reader, plate_crop, debug=debug)

    return plate, score
