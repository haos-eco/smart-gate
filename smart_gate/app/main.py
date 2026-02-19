import os
import time
import cv2

from utils import get_options, validate_model, capture_and_recognize, fuzzy_match


def main():
    opt = get_options()

    # Parse ROI
    roi_raw = opt.get("roi", "0.0,0.0,1.0,1.0")
    roi = [float(x.strip()) for x in roi_raw.split(",")] if isinstance(roi_raw, str) else roi_raw

    # Configuration
    motion_entity = opt["motion_entity"]
    camera_entity = opt["camera_entity"]
    gate_switch = opt["gate_switch"]
    conf = float(opt.get("confidence", 0.35))
    cooldown = int(opt.get("cooldown_sec", 30))
    model_path = opt.get("model_path", "/config/www/smart_gate/models/yolo/model.onnx")
    snapshot_path = opt.get("snapshot_path", "/config/www/smart_gate/snapshot/latest.jpg")
    history_dir = opt.get("history_dir", "/config/www/smart_gate/snapshot/history")
    debug_crop_path = opt.get("debug_path", "/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg")
    keep_history = bool(opt.get("keep_history", False))
    gpu = bool(opt.get("gpu", False))
    debug = bool(opt.get("debug", False))

    # Build plate -> person_entity map
    # Supports both formats:
    #   - {plate: "CB234YC", person_entity: "person.andrea"}  (with GPS check on fuzzy)
    #   - "CB234YC"  (legacy plain string, no GPS check)
    plate_to_person = {}
    for entry in opt.get("allowed_plates", []):
        if isinstance(entry, dict):
            p = entry.get("plate", "").strip().upper()
            person = entry.get("person_entity", "")
            if p:
                plate_to_person[p] = person
        else:
            plate_to_person[entry.strip().upper()] = ""

    allowed_plates = set(plate_to_person.keys())

    if not validate_model(model_path):
        return

    print("Loading YOLO model...")
    from detection import load_model
    sess, inp, out = load_model(model_path)

    print("Loading EasyOCR... (first run may take 2-3 minutes)")
    import easyocr
    reader = easyocr.Reader(['en'], gpu)

    print("Initializing AI Super-Resolution...")
    from image_processing import get_sr_model
    sr_model = get_sr_model()

    if sr_model is not None:
        print("✅ AI Super-Resolution: ENABLED (EDSR 2x)")
    else:
        print("ℹ️  AI Super-Resolution: DISABLED (using bicubic fallback)")

    last_open = 0.0
    last_motion = ""

    print("Smart Gate started. Watching motion:", motion_entity)
    if debug:
        print(f"DEBUG - Confidence: {conf}, ROI: {roi}, Allowed: {allowed_plates}")
        print(f"DEBUG - Crop path: {debug_crop_path}")

    while True:
        try:
            from homeassistant import get_state, switch_on

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
                    reader, roi, conf, debug,
                    debug_crop_path=debug_crop_path,
                    attempt_number=i+1
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

            from collections import Counter
            plate_counts = Counter(attempts)
            most_common = plate_counts.most_common(1)[0]
            plate, count = most_common

            print(f"📊 Consensus: '{plate}' ({count}/{len(attempts)} attempts)")

            # Security: require at least 2/3 consensus
            if count < 2:
                print(f"⚠️  No consensus (need 2/3), gate stays closed")
                time.sleep(1)
                continue

            if keep_history:
                os.makedirs(history_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                img = cv2.imread(snapshot_path)
                if img is not None:
                    cv2.imwrite(os.path.join(history_dir, f"{ts}_{plate}.jpg"), img)

            matched, distance = fuzzy_match(plate, list(allowed_plates), max_distance=2)

            if matched is None:
                if debug:
                    print(f"❌ Plate '{plate}' not in whitelist (no fuzzy match)")
                time.sleep(2)
                continue

            if distance == 0:
                # Exact match — open immediately, GPS not required
                print(f"✅ Exact match '{plate}' → gate opening")
                switch_on(gate_switch)
                last_open = now

            else:
                # Fuzzy match — check the person specifically linked to this plate
                person_entity = plate_to_person.get(matched, "")

                if not person_entity:
                    print(f"⛔ Fuzzy match '{plate}' ≈ '{matched}' (distance: {distance}) but no person_entity configured — gate stays closed")
                    time.sleep(2)
                    continue

                print(f"🔍 Fuzzy match '{plate}' ≈ '{matched}' (distance: {distance}) — checking {person_entity}...")
                try:
                    person_state = get_state(person_entity)
                except Exception as e:
                    print(f"⚠️  Could not get state for {person_entity}: {e} — gate stays closed")
                    time.sleep(2)
                    continue

                if person_state == "home":
                    print(f"✅ Fuzzy match + {person_entity} home → gate opening (read '{plate}', matched '{matched}')")
                    switch_on(gate_switch)
                    last_open = now
                else:
                    print(f"⛔ Fuzzy match '{plate}' ≈ '{matched}' but {person_entity} not home — gate stays closed")

            time.sleep(2)

        except Exception as e:
            print(f"ERROR: {repr(e)}")
            if debug:
                import traceback
                traceback.print_exc()
            time.sleep(2)

if __name__ == "__main__":
    main()
