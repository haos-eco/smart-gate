import os
import time
import cv2

from utils import get_options, is_complete_plate, validate_model, capture_and_recognize, fuzzy_match


def main():
    opt = get_options()

    # Parse ROI
    roi_raw = opt.get("roi", "0.0,0.0,1.0,1.0")
    roi = [float(x.strip()) for x in roi_raw.split(",")] if isinstance(roi_raw, str) else roi_raw

    motion_entity = opt["motion_entity"]
    camera_entity = opt["camera_entity"]
    gate_switch = opt["gate_switch"]
    conf = float(opt.get("confidence", 0.5))
    cooldown = int(opt.get("cooldown_sec", 60))
    min_yolo_score = float(opt.get("min_yolo_score", 0.65))
    min_ocr_confidence = float(opt.get("min_ocr_confidence", 0.75))
    model_path = opt.get("model_path", "/config/www/smart_gate/models/yolo/model.onnx")
    snapshot_path = opt.get("snapshot_path", "/config/www/smart_gate/snapshot/latest.jpg")
    history_dir = opt.get("history_dir", "/config/www/smart_gate/snapshot/history")
    notify_devices = opt.get("notify_devices", [])
    visitor_stop_sec = int(opt.get("visitor_stop_sec", 5))
    notification_sound = opt.get("notification_sound", "default")
    debug_crop_path = opt.get("debug_path", "/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg")
    keep_history = bool(opt.get("keep_history", False))
    gpu = bool(opt.get("gpu", False))
    debug = bool(opt.get("debug", False))

    if isinstance(notify_devices, str):
        notify_services = [notify_devices] if notify_devices else []
    else:
        notify_services = [s for s in notify_devices if s]

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
    motion_off_since = None       # timestamp when motion went OFF
    visitor_notified = False      # avoid sending multiple notifications per stop event
    notification_thread = None    # background thread listening for open action

    print("Smart Gate started. Watching motion:", motion_entity)
    if debug:
        print(f"DEBUG - YOLO conf threshold: {conf}, min_yolo_score: {min_yolo_score}, min_ocr_confidence: {min_ocr_confidence}")
        print(f"DEBUG - ROI: {roi}, Allowed plates: {allowed_plates}")
        if notify_services:
            print(f"DEBUG - Visitor notification: {notify_devices}, stop threshold: {visitor_stop_sec}s")
        else:
            print("DEBUG - Visitor notification: disabled (notify_services not set)")



    while True:
        try:
            from homeassistant import get_state, switch_on, send_visitor_notification, camera_snapshot

            state = get_state(motion_entity)

            # Track motion state transitions
            if state != last_motion:
                print(f"State changed: {last_motion} -> {state}")

                if state == "off":
                    # Vehicle stopped or left — start stop timer
                    motion_off_since = time.time()
                    visitor_notified = False
                elif state == "on":
                    # New motion — reset stop timer and notification flag
                    motion_off_since = None
                    visitor_notified = False

                last_motion = state

            # --- Visitor notification: vehicle stopped for visitor_stop_sec ---
            if (
                    notify_devices
                    and state == "off"
                    and motion_off_since is not None
                    and not visitor_notified
                    and (time.time() - motion_off_since) >= visitor_stop_sec
            ):
                print(f"🔔 Vehicle stopped for {visitor_stop_sec}s — sending visitor notification...")
                try:
                    camera_snapshot(camera_entity, snapshot_path)
                    send_visitor_notification(notify_services, snapshot_path, camera_entity, notification_sound)
                    visitor_notified = True
                    print("🔔 Visitor notification sent")

                    # Start background thread to listen for open gate action
                    if notification_thread is None or not notification_thread.is_alive():
                        from notifications import start_notification_listener
                        notification_thread = start_notification_listener(gate_switch, debug)
                except Exception as e:
                    print(f"⚠️  Failed to send visitor notification: {e}")

            # --- License plate recognition: only when motion is ON ---
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

            # 3 attempts — prefer complete plates (AA123AA), then best combined score
            best_plate = None
            best_yolo = 0.0
            best_ocr = 0.0
            best_is_complete = False

            for i in range(3):
                plate, yolo_score, ocr_conf = capture_and_recognize(
                    camera_entity, snapshot_path, sess, inp, out,
                    reader, roi, conf, debug,
                    debug_crop_path=debug_crop_path,
                    attempt_number=i+1
                )

                if plate:
                    print(f"  Attempt {i+1}/3: '{plate}' (YOLO: {yolo_score:.3f}, OCR: {ocr_conf:.3f})")
                    is_complete = is_complete_plate(plate)
                    combined = yolo_score + ocr_conf
                    best_combined = best_yolo + best_ocr
                    # A complete plate always beats a partial one
                    # Among equal completeness, pick best combined score
                    if (is_complete and not best_is_complete) or \
                            (is_complete == best_is_complete and combined > best_combined):
                        best_plate = plate
                        best_yolo = yolo_score
                        best_ocr = ocr_conf
                        best_is_complete = is_complete
                else:
                    print(f"  Attempt {i+1}/3: No plate detected")

                if i < 2:
                    time.sleep(0.5)

            if not best_plate:
                print("❌ No plates detected in any attempt")
                time.sleep(1)
                continue

            print(f"📊 Best detection: '{best_plate}' (YOLO: {best_yolo:.3f}, OCR: {best_ocr:.3f})")

            # Quality check: require minimum scores to proceed
            if best_yolo < min_yolo_score:
                print(f"⚠️  YOLO score too low ({best_yolo:.3f} < {min_yolo_score}) — gate stays closed")
                time.sleep(1)
                continue

            if best_ocr < min_ocr_confidence:
                print(f"⚠️  OCR confidence too low ({best_ocr:.3f} < {min_ocr_confidence}) — gate stays closed")
                time.sleep(1)
                continue

            if keep_history:
                os.makedirs(history_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                img = cv2.imread(snapshot_path)
                if img is not None:
                    cv2.imwrite(os.path.join(history_dir, f"{ts}_{best_plate}.jpg"), img)

            matched, distance = fuzzy_match(best_plate, list(allowed_plates), max_distance=2)

            if matched is None:
                if debug:
                    print(f"❌ Plate '{best_plate}' not in whitelist (no fuzzy match)")
                time.sleep(2)
                continue

            if distance == 0:
                # Exact match — open immediately, GPS not required
                print(f"✅ Exact match '{best_plate}' → gate opening")
                switch_on(gate_switch)
                last_open = now

            else:
                # Fuzzy match — check the person specifically linked to this plate
                person_entity = plate_to_person.get(matched, "")

                if not person_entity:
                    print(f"⛔ Fuzzy match '{best_plate}' ≈ '{matched}' (distance: {distance}) but no person_entity configured — gate stays closed")
                    time.sleep(2)
                    continue

                print(f"🔍 Fuzzy match '{best_plate}' ≈ '{matched}' (distance: {distance}) — checking {person_entity}...")
                try:
                    person_state = get_state(person_entity)
                except Exception as e:
                    print(f"⚠️  Could not get state for {person_entity}: {e} — gate stays closed")
                    time.sleep(2)
                    continue

                if person_state == "home":
                    print(f"✅ Fuzzy match + {person_entity} home → gate opening (read '{best_plate}', matched '{matched}')")
                    switch_on(gate_switch)
                    last_open = now
                else:
                    print(f"⛔ Fuzzy match '{best_plate}' ≈ '{matched}' but {person_entity} not home — gate stays closed")

            time.sleep(2)

        except Exception as e:
            print(f"ERROR: {repr(e)}")
            if debug:
                import traceback
                traceback.print_exc()
            time.sleep(2)

if __name__ == "__main__":
    main()
