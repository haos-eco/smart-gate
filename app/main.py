from collections import Counter
import os
import time
import cv2

from utils import get_options, capture_and_recognize
from homeassistant import get_state, camera_snapshot, switch_on
from detection import load_model

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

    if not validate_model(model_path):
        return

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

            print("Motion detected! Multi-attempt recognition...")

            # Multi-attempt: 3 captures with 0.5s pause
            attempts = []
            for i in range(3):
                plate, score = capture_and_recognize(
                    camera_entity, snapshot_path, sess, inp, out,
                    reader, roi, conf, debug
                )

                if plate:
                    attempts.append(plate)
                    print(f"  Attempt {i+1}/3: '{plate}' (YOLO score: {score:.3f})")
                else:
                    print(f"  Attempt {i+1}/3: No plate detected")

                if i < 2:  # Don't wait after last attempt
                    time.sleep(0.5)

            if not attempts:
                print("❌ No plates detected in any attempt")
                time.sleep(1)
                continue

            # Consensus: require 2/3 agreement
            plate_counts = Counter(attempts)
            most_common = plate_counts.most_common(1)[0]
            plate, count = most_common

            print(f"📊 Consensus: '{plate}' ({count}/{len(attempts)} attempts)")

            # Security: require at least 2/3 consensus
            if count < 2:
                print(f"⚠️  No consensus (need 2/3), gate stays closed")
                time.sleep(1)
                continue

            # Save history if enabled
            if keep_history:
                os.makedirs(history_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                img = cv2.imread(snapshot_path)
                if img is not None:
                    cv2.imwrite(os.path.join(history_dir, f"{ts}_{plate}.jpg"), img)

            # Exact match on whitelist
            if plate in allowed:
                switch_on(gate_switch)
                last_open = now
                print(f"✅ Gate opened for: {plate}")
            elif debug:
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
