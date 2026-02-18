import os
import time
import cv2

from utils import get_options, ensure_dir
from homeassistant import get_state, camera_snapshot, switch_on
from image_processing import (
    apply_roi,
    is_infrared,
    remove_plate_border,
    fix_overexposure
)
from ocr import ocr_plate
from detection import load_model, detect_plates

def validate_model(model_path: str) -> bool:
    """Validate model file exists and has correct size"""
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

def main():
    opt = get_options()

    # Parse ROI
    roi_raw = opt.get("roi", "0.0,0.0,1.0,1.0")
    roi = [float(x.strip()) for x in roi_raw.split(",")] if isinstance(roi_raw, str) else roi_raw

    # Configuration
    motion_entity = opt["motion_entity"]
    camera_entity = opt["camera_entity"]
    gate_switch = opt["gate_switch"]
    allowed = {p.strip().upper() for p in opt.get("allowed_plates", [])}
    conf = float(opt.get("confidence", 0.35))
    cooldown = int(opt.get("cooldown_sec", 30))
    model_path = opt.get("model_path", "/config/www/smart_gate/model.onnx")
    snapshot_path = opt.get("snapshot_path", "/config/www/smart_gate/snapshot/latest.jpg")
    history_dir = opt.get("history_dir", "/config/www/smart_gate/snapshot/history")
    debug_crop_path = opt.get("debug_path", "/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg")
    keep_history = bool(opt.get("keep_history", False))
    debug = bool(opt.get("debug", False))

    # Validate model
    if not validate_model(model_path):
        return

    # Load models
    print("Loading YOLO model...")
    sess, inp, out = load_model(model_path)

    print("Loading EasyOCR... (first run may take 2-3 minutes)")
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)

    # State
    last_open = 0.0
    last_motion = ""

    print("Smart Gate started. Watching motion:", motion_entity)
    if debug:
        print(f"DEBUG - Confidence: {conf}, ROI: {roi}, Allowed: {allowed}")

    # Main loop
    while True:
        try:
            # Check motion sensor
            state = get_state(motion_entity)
            if state != last_motion:
                print(f"State changed: {last_motion} -> {state}")
                last_motion = state

            if state != "on":
                time.sleep(0.5)
                continue

            # Cooldown check
            now = time.time()
            if now - last_open < cooldown:
                if debug:
                    print(f"In cooldown ({now - last_open:.1f}s / {cooldown}s)")
                time.sleep(0.5)
                continue

            print("Motion detected! Capturing snapshot...")

            # Capture and load image
            ensure_dir(snapshot_path)
            camera_snapshot(camera_entity, snapshot_path)

            img = cv2.imread(snapshot_path)
            if img is None:
                time.sleep(1)
                continue

            img_roi = apply_roi(img, roi)

            if debug:
                print(f"DEBUG - Image: {img.shape[1]}x{img.shape[0]}, ROI: {img_roi.shape[1]}x{img_roi.shape[0]}")

            # YOLO detection
            print(f"Running YOLO (conf={conf})...")
            boxes = detect_plates(sess, inp, out, img_roi, conf=conf, debug=debug)
            print(f"YOLO found {len(boxes)} boxes")

            if debug and boxes:
                for i, (x1, y1, x2, y2, s) in enumerate(boxes[:3]):
                    print(f"  Box {i}: score={s:.2f}, size={x2-x1}x{y2-y1}")

            if not boxes:
                print("No plates detected")
                time.sleep(1)
                continue

            # Get best box
            boxes.sort(key=lambda b: b[4], reverse=True)
            x1, y1, x2, y2, score = boxes[0]
            plate_crop = img_roi[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

            if plate_crop.size == 0:
                time.sleep(1)
                continue

            # Preprocessing
            plate_crop = remove_plate_border(plate_crop)

            ir_detected, saturation = is_infrared(plate_crop)
            if ir_detected:
                plate_crop = fix_overexposure(plate_crop)
                if debug:
                    print(f"IR detected (sat={saturation:.1f}) - fixing exposure")

            if debug:
                ensure_dir(debug_crop_path)
                cv2.imwrite(debug_crop_path, plate_crop)
                print(f"Crop saved: {debug_crop_path}")

            # OCR
            plate = ocr_plate(reader, plate_crop, debug=debug)
            print(f"Detected: {plate} (score: {score:.3f})")

            # Save history
            if keep_history:
                os.makedirs(history_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(history_dir, f"{ts}_{plate or 'NONE'}.jpg"), img)

            # Check whitelist and open gate
            if plate and plate in allowed:
                switch_on(gate_switch)
                last_open = now
                print(f"✅ Gate opened for: {plate}")
            elif debug and plate:
                print(f"❌ Plate '{plate}' not in whitelist")

            time.sleep(2)

        except Exception as e:
            print(f"ERROR: {repr(e)}")
            if debug:
                import traceback
                traceback.print_exc()
            time.sleep(2)

if __name__ == "__main__":
    main()
