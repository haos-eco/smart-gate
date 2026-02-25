import os
import threading
import time
import cv2

from access_log import log_event
from constants import CLEANUP_INTERVAL
from dashboard_installer import install_dashboard
from watchers import start_config_watcher
from snapshot_processing import annotate_snapshot
from motion import apply_privacy_mask, validate_motion_outside_zones
from utils import (
    get_options,
    is_complete_plate,
    validate_model,
    capture_frame,
    recognize_frame,
    fuzzy_match,
    cleanup_history,
)


def main():
    install_dashboard()
    start_config_watcher()

    opt = get_options()

    # Parse ROI
    roi_raw = opt.get("roi", "0.0,0.0,1.0,1.0")
    roi = (
        [float(x.strip()) for x in roi_raw.split(",")]
        if isinstance(roi_raw, str)
        else roi_raw
    )

    motion_entity = opt["motion_entity"]
    camera_entity = opt["camera_entity"]
    gate_switch = opt["gate_switch"]
    conf = float(opt.get("confidence", 0.5))
    cooldown = int(opt.get("cooldown_sec", 60))
    min_yolo_score = float(opt.get("min_yolo_score", 0.65))
    min_ocr_confidence = float(opt.get("min_ocr_confidence", 0.75))
    model_path = opt.get("model_path", "/config/www/smart_gate/models/yolo/model.onnx")
    snapshot_path = opt.get(
        "snapshot_path", "/config/www/smart_gate/snapshot/latest.jpg"
    )
    history_dir = opt.get("history_dir", "/config/www/smart_gate/snapshot/history")
    logs_dir = opt.get("logs_dir", "/config/www/smart_gate/snapshot/logs")
    notify_services = opt.get("notify_services", [])
    visitor_stop_sec = int(opt.get("visitor_stop_sec", 5))
    notification_sound = opt.get("notification_sound", "default")
    debug_crop_path = opt.get(
        "debug_path", "/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg"
    )
    keep_history = bool(opt.get("keep_history", False))
    keep_history_days = int(opt.get("keep_history_days", 30))
    notify_on_failure = bool(opt.get("notify_on_failure", False))
    gpu = bool(opt.get("gpu", False))
    debug = bool(opt.get("debug", False))

    if isinstance(notify_services, str):
        _notify_services = [notify_services] if notify_services else []
    else:
        _notify_services = [s for s in notify_services if s]

    # Build plate -> person_entity map
    # Supports both formats:
    #   - {plate: "CB234YC", person_entity: "person.andrea"}  (GPS check on fuzzy)
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

    os.makedirs(logs_dir, exist_ok=True)

    print("Loading YOLO model...")
    from detection import load_model

    sess, inp, out = load_model(model_path)

    # TrOCR is the primary OCR engine.
    # EasyOCR is loaded only if TrOCR fails to initialize — never both at once.
    from trocr import load_trocr, trocr_infer

    trocr_ok = load_trocr()

    if trocr_ok:
        print("✅ TrOCR ready — EasyOCR will not be loaded")
        reader = None
        ocr_engine = "trocr"
    else:
        print("⚠️  TrOCR unavailable — loading EasyOCR fallback...")
        import easyocr

        reader = easyocr.Reader(["en"], gpu)
        ocr_engine = "easyocr"

    print(f"OCR engine: {ocr_engine}")

    print("Initializing AI Super-Resolution...")
    from image_processing import get_sr_model

    sr_model = get_sr_model()

    if sr_model is not None:
        print("✅ AI Super-Resolution: ENABLED (EDSR 2x)")
    else:
        print("ℹ️  AI Super-Resolution: DISABLED (using bicubic fallback)")

    last_open = 0.0
    last_cleanup = 0.0
    last_motion = ""
    motion_off_since = None  # timestamp when motion went OFF
    visitor_notified = False  # avoid sending multiple notifications per stop event
    notification_thread = None  # background thread listening for open action
    vehicle_detected = False  # True if YOLO found a plate during last motion ON session
    last_vehicle_snapshot = None  # path of snapshot taken when vehicle was detected

    print("Smart Gate started. Watching motion:", motion_entity)
    if debug:
        print(f"DEBUG - OCR engine: {ocr_engine}")
        print(
            f"DEBUG - YOLO conf threshold: {conf}, min_yolo_score: {min_yolo_score}, min_ocr_confidence: {min_ocr_confidence}"
        )
        print(f"DEBUG - ROI: {roi}, Allowed plates: {allowed_plates}")
        print(f"DEBUG - History retention: {keep_history_days} days")
        print(f"DEBUG - Logs dir: {logs_dir}, History dir: {history_dir}")
        if _notify_services:
            print(
                f"DEBUG - Visitor notification: {_notify_services}, stop threshold: {visitor_stop_sec}s"
            )
        else:
            print("DEBUG - Visitor notification: disabled (notify_services not set)")

    while True:
        try:
            from homeassistant import get_state, switch_on
            from notifications import send_visitor_notification

            if time.time() - last_cleanup > CLEANUP_INTERVAL:
                cleanup_history(history_dir, logs_dir, keep_history_days)
                last_cleanup = time.time()

            state = get_state(motion_entity)

            # ── Motion state transitions ───────────────────────────────────
            if state != last_motion:
                print(f"State changed: {last_motion} -> {state}")

                if state == "off":
                    motion_off_since = time.time()
                    visitor_notified = False
                elif state == "on":
                    motion_off_since = None
                    visitor_notified = False
                    vehicle_detected = False
                    last_vehicle_snapshot = None

                last_motion = state

            # ── Visitor notification: vehicle stopped for visitor_stop_sec ─
            if (
                _notify_services
                and state == "off"
                and motion_off_since is not None
                and not visitor_notified
                and vehicle_detected
                and (time.time() - motion_off_since) >= visitor_stop_sec
            ):
                print(
                    f"🔔 Vehicle stopped for {visitor_stop_sec}s — sending visitor notification..."
                )
                try:
                    notification_snapshot = last_vehicle_snapshot or snapshot_path
                    send_visitor_notification(
                        _notify_services, notification_snapshot, notification_sound
                    )
                    visitor_notified = True
                    print("🔔 Visitor notification sent")

                    if (
                        notification_thread is None
                        or not notification_thread.is_alive()
                    ):
                        from notifications import start_notification_listener

                        notification_thread = start_notification_listener(
                            gate_switch, debug
                        )
                except Exception as e:
                    print(f"⚠️  Failed to send visitor notification: {e}")

            # ── License plate recognition: only when motion is ON ──────────
            if state != "on":
                time.sleep(0.1)
                continue

            # Cooldown check
            now = time.time()
            if now - last_open < cooldown:
                if debug:
                    print(f"In cooldown ({now - last_open:.1f}s / {cooldown}s)")
                time.sleep(0.5)
                continue

            print("Motion detected! Multi-attempt recognition...")

            # 3 attempts — priority: exact whitelist match > complete plate > best combined score
            best_plate = None
            best_bbox = None  # (x1, y1, x2, y2) of the best detection
            best_yolo = 0.0
            best_ocr = 0.0
            best_is_complete = False
            best_is_exact = False
            early_exit = False
            prefetched = [None]

            def _fetch(dest):
                try:
                    frame = capture_frame(camera_entity, snapshot_path)
                    if frame is not None:
                        # Apply privacy mask and overwrite snapshot on disk (async)
                        # so visitor notifications always use the masked image
                        frame = apply_privacy_mask(frame)
                        threading.Thread(
                            target=cv2.imwrite, args=(snapshot_path, frame), daemon=True
                        ).start()
                    dest[0] = frame
                except Exception as e:
                    print(f"  ⚠️  Capture error: {e}")
                    dest[0] = None

            # Kick off first prefetch immediately — no extra capture for validation
            fetch_thread = threading.Thread(
                target=_fetch, args=(prefetched,), daemon=True
            )
            fetch_thread.start()

            for i in range(3):
                fetch_thread.join()
                img = prefetched[0]

                # Start next prefetch in parallel while we process the current frame
                if i < 2:
                    prefetched = [None]
                    fetch_thread = threading.Thread(
                        target=_fetch, args=(prefetched,), daemon=True
                    )
                    fetch_thread.start()

                # On attempt 0: validate motion outside masked zones using the frame
                # already fetched — zero extra I/O, runs concurrently with next prefetch.
                if i == 0 and not validate_motion_outside_zones(img):
                    if debug:
                        print(
                            "DEBUG - Motion ignored: movement only inside masked zones"
                        )
                    break

                plate, yolo_score, ocr_conf, det_snapshot, bbox = recognize_frame(
                    img,
                    sess,
                    inp,
                    out,
                    reader,
                    roi,
                    conf,
                    debug,
                    logs_dir=logs_dir,
                    label=f"detection_attempt{i+1}",
                    debug_crop_path=debug_crop_path,
                    attempt_number=i + 1,
                    ocr_engine=ocr_engine,
                    trocr_infer_fn=trocr_infer if trocr_ok else None,
                )

                if plate:
                    vehicle_detected = True
                    if det_snapshot:
                        last_vehicle_snapshot = det_snapshot
                    print(
                        f"  Attempt {i+1}/3: '{plate}' (YOLO: {yolo_score:.3f}, OCR: {ocr_conf:.3f})"
                    )
                    is_complete = is_complete_plate(plate)
                    is_exact = plate in allowed_plates
                    combined = yolo_score + ocr_conf
                    best_combined = best_yolo + best_ocr
                    # Exact whitelist match always wins
                    # Then complete plate over partial
                    # Then best combined score as tiebreaker
                    if (
                        (is_exact and not best_is_exact)
                        or (
                            is_exact == best_is_exact
                            and is_complete
                            and not best_is_complete
                        )
                        or (
                            is_exact == best_is_exact
                            and is_complete == best_is_complete
                            and combined > best_combined
                        )
                    ):
                        best_plate = plate
                        best_bbox = bbox
                        best_yolo = yolo_score
                        best_ocr = ocr_conf
                        best_is_complete = is_complete
                        best_is_exact = is_exact

                    # Early exit 1: exact whitelist match
                    if is_exact:
                        print(f"  ⚡ Early exit: exact match at attempt {i+1}/3")
                        early_exit = True
                        break

                    # Early exit 2: very high confidence complete plate
                    if is_complete and yolo_score >= 0.90 and ocr_conf >= 0.92:
                        print(
                            f"  ⚡ Early exit: high-confidence plate at attempt {i+1}/3 "
                            f"(YOLO: {yolo_score:.3f}, OCR: {ocr_conf:.3f})"
                        )
                        early_exit = True
                        break
                else:
                    print(f"  Attempt {i+1}/3: No plate detected")

            # ── No plate detected ──────────────────────────────────────────
            if not best_plate:
                print("❌ No plates detected in any attempt")
                if notify_on_failure and _notify_services and vehicle_detected:
                    try:
                        fail_snapshot = last_vehicle_snapshot or snapshot_path
                        annotated = annotate_snapshot(
                            fail_snapshot, "UNREADABLE", None, 0.0, 0.0, "unreadable"
                        )
                        send_visitor_notification(
                            _notify_services,
                            annotated,
                            notification_sound,
                            title="⚠️ SmartGate: targa illeggibile",
                            message="Veicolo rilevato ma targa non riconosciuta.",
                        )
                        print("🔔 Failure notification sent (unreadable plate)")
                    except Exception as e:
                        print(f"⚠️  Failed to send failure notification: {e}")
                time.sleep(1)
                continue

            if early_exit:
                print(
                    f"📊 Best detection: '{best_plate}' (YOLO: {best_yolo:.3f}, OCR: {best_ocr:.3f}) [early exit]"
                )
            else:
                print(
                    f"📊 Best detection: '{best_plate}' (YOLO: {best_yolo:.3f}, OCR: {best_ocr:.3f})"
                )

            # ── Quality gates (skipped for exact whitelist matches) ────────
            if not best_is_exact:
                if best_yolo < min_yolo_score:
                    reason = f"YOLO score too low ({best_yolo:.3f})"
                    print(f"⚠️  {reason} — gate stays closed")
                    snap = last_vehicle_snapshot or snapshot_path
                    annotated = annotate_snapshot(
                        snap, best_plate, best_bbox, best_yolo, best_ocr, "rejected"
                    )
                    log_event(
                        best_plate,
                        "rejected",
                        annotated,
                        "none",
                        best_yolo,
                        best_ocr,
                        reason=reason,
                    )
                    if notify_on_failure and _notify_services:
                        try:
                            send_visitor_notification(
                                _notify_services,
                                annotated,
                                notification_sound,
                                title=f"⚠️ SmartGate: targa rifiutata ({best_plate})",
                                message=f"Rilevamento non affidabile. {reason}.",
                            )
                            print("🔔 Failure notification sent (low YOLO score)")
                        except Exception as e:
                            print(f"⚠️  Failed to send failure notification: {e}")
                    time.sleep(1)
                    continue

                if best_ocr < min_ocr_confidence:
                    reason = f"OCR confidence too low ({best_ocr:.3f})"
                    print(f"⚠️  {reason} — gate stays closed")
                    snap = last_vehicle_snapshot or snapshot_path
                    annotated = annotate_snapshot(
                        snap, best_plate, best_bbox, best_yolo, best_ocr, "rejected"
                    )
                    log_event(
                        best_plate,
                        "rejected",
                        annotated,
                        "none",
                        best_yolo,
                        best_ocr,
                        reason=reason,
                    )
                    if notify_on_failure and _notify_services:
                        try:
                            send_visitor_notification(
                                _notify_services,
                                annotated,
                                notification_sound,
                                title=f"⚠️ SmartGate: targa rifiutata ({best_plate})",
                                message=f"OCR non affidabile. {reason}.",
                            )
                            print("🔔 Failure notification sent (low OCR confidence)")
                        except Exception as e:
                            print(f"⚠️  Failed to send failure notification: {e}")
                    time.sleep(1)
                    continue

            if keep_history:
                os.makedirs(history_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                img = cv2.imread(snapshot_path)
                if img is not None:
                    cv2.imwrite(
                        os.path.join(history_dir, f"{ts}_{best_plate}.jpg"), img
                    )

            matched, distance = fuzzy_match(
                best_plate, list(allowed_plates), max_distance=2
            )

            # ── Not in whitelist ───────────────────────────────────────────
            if matched is None:
                if debug:
                    print(f"❌ Plate '{best_plate}' not in whitelist (no fuzzy match)")
                snap = last_vehicle_snapshot or snapshot_path
                annotated = annotate_snapshot(
                    snap, best_plate, best_bbox, best_yolo, best_ocr, "unknown"
                )
                log_event(
                    best_plate,
                    "unknown",
                    annotated,
                    "none",
                    best_yolo,
                    best_ocr,
                    reason="Not in whitelist",
                )
                if notify_on_failure and _notify_services:
                    try:
                        send_visitor_notification(
                            _notify_services,
                            annotated,
                            notification_sound,
                            title=f"🚗 SmartGate: targa sconosciuta ({best_plate})",
                            message="Veicolo non in whitelist.",
                        )
                        print("🔔 Failure notification sent (unknown plate)")
                    except Exception as e:
                        print(f"⚠️  Failed to send failure notification: {e}")
                time.sleep(2)
                continue

            # ── Exact match ────────────────────────────────────────────────
            if distance == 0:
                snap = last_vehicle_snapshot or snapshot_path
                annotated = annotate_snapshot(
                    snap, best_plate, best_bbox, best_yolo, best_ocr, "opened"
                )
                last_vehicle_snapshot = annotated
                switch_on(gate_switch)
                last_open = now
                print(f"✅ Exact match '{best_plate}' → gate opening")
                threading.Thread(
                    target=log_event,
                    args=(
                        best_plate,
                        "opened",
                        annotated,
                        "exact match",
                        best_yolo,
                        best_ocr,
                    ),
                    kwargs={"matched_plate": matched, "reason": "Plate in whitelist"},
                    daemon=True,
                ).start()

            # ── Fuzzy match ────────────────────────────────────────────────
            else:
                person_entity = plate_to_person.get(matched, "")

                if not person_entity:
                    print(
                        f"⛔ Fuzzy match '{best_plate}' ≈ '{matched}' (distance: {distance}) but no person_entity configured — gate stays closed"
                    )
                    log_event(
                        best_plate,
                        "rejected",
                        last_vehicle_snapshot or snapshot_path,
                        "fuzzy",
                        best_yolo,
                        best_ocr,
                        matched_plate=matched,
                        reason="No person_entity configured",
                    )
                    time.sleep(2)
                    continue

                print(
                    f"🔍 Fuzzy match '{best_plate}' ≈ '{matched}' (distance: {distance}) — checking {person_entity}..."
                )
                try:
                    person_state = get_state(person_entity)
                except Exception as e:
                    print(
                        f"⚠️  Could not get state for {person_entity}: {e} — gate stays closed"
                    )
                    time.sleep(2)
                    continue

                if person_state == "home":
                    snap = last_vehicle_snapshot or snapshot_path
                    annotated = annotate_snapshot(
                        snap, best_plate, best_bbox, best_yolo, best_ocr, "opened"
                    )
                    last_vehicle_snapshot = annotated
                    print(
                        f"✅ Fuzzy match + {person_entity} home → gate opening (read '{best_plate}', matched '{matched}')"
                    )
                    switch_on(gate_switch)
                    log_event(
                        best_plate,
                        "opened",
                        annotated,
                        "fuzzy",
                        best_yolo,
                        best_ocr,
                        matched_plate=matched,
                    )
                    last_open = now
                else:
                    snap = last_vehicle_snapshot or snapshot_path
                    annotated = annotate_snapshot(
                        snap, best_plate, best_bbox, best_yolo, best_ocr, "rejected"
                    )
                    print(
                        f"⛔ Fuzzy match '{best_plate}' ≈ '{matched}' but {person_entity} not home — gate stays closed"
                    )
                    log_event(
                        best_plate,
                        "rejected",
                        annotated,
                        "fuzzy",
                        best_yolo,
                        best_ocr,
                        matched_plate=matched,
                        reason=f"{person_entity} not home",
                    )
                    if notify_on_failure and _notify_services:
                        try:
                            send_visitor_notification(
                                _notify_services,
                                annotated,
                                notification_sound,
                                title=f"⚠️ SmartGate: accesso negato ({best_plate})",
                                message=f"Fuzzy match con '{matched}' ma {person_entity} non è a casa.",
                            )
                            print("🔔 Failure notification sent (person not home)")
                        except Exception as e:
                            print(f"⚠️  Failed to send failure notification: {e}")

            time.sleep(2)

        except Exception as e:
            print(f"ERROR: {repr(e)}")
            if debug:
                import traceback

                traceback.print_exc()
            time.sleep(2)


if __name__ == "__main__":
    main()
