import os
import re
import time
import json
import cv2
import numpy as np
import easyocr
import requests

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")
HASS_URL = "http://supervisor/core/api"
HEADERS = {"Authorization": f"Bearer {SUPERVISOR_TOKEN}", "Content-Type": "application/json"}

def get_options():
    # Supervisor injects options into /data/options.json
    with open("/data/options.json", "r", encoding="utf-8") as f:
        return json.load(f)

def ha_get_state(entity_id: str) -> str:
    r = requests.get(f"{HASS_URL}/states/{entity_id}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json().get("state", "")

def ha_call(service: str, data: dict):
    domain, svc = service.split(".", 1)
    r = requests.post(f"{HASS_URL}/services/{domain}/{svc}", headers=HEADERS, json=data, timeout=15)
    r.raise_for_status()
    return r.json()

def camera_snapshot(camera_entity: str, path: str):
    ha_call("camera.snapshot", {"entity_id": camera_entity, "filename": path})

def switch_on(entity_id: str):
    ha_call("switch.turn_on", {"entity_id": entity_id})

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def is_infrared(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    is_ir = mean_saturation < 20
    return is_ir, mean_saturation

def remove_plate_border(img_crop):
    h, w = img_crop.shape[:2]
    margin_h = int(h * 0.18)
    margin_w = int(w * 0.08)
    return img_crop[margin_h:h-margin_h, margin_w:w-margin_w]

def fix_overexposure(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def crop_white_area(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    col_sums = np.sum(mask, axis=0)
    threshold = img_bgr.shape[0] * 0.3 * 255

    white_cols = np.where(col_sums > threshold)[0]

    if len(white_cols) > 0:
        x_start = white_cols[0]
        x_end = white_cols[-1]
        return img_bgr[:, x_start:x_end]

    return img_bgr

def extract_plate_pattern(text):
    """Estrae pattern AA123AA da stringhe più lunghe"""
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
    letter_fixes = {'0': 'O', '1': 'I', '4': 'A', '8': 'B'}
    number_fixes = {'O': '0', 'I': '1', 'Z': '4', 'S': '5', 'B': '8'}

    # IT format: AA123AA
    if len(plate_text) == 7:
        result = list(plate_text)

        # 0,1 positions should be letters
        for i in [0, 1]:
            if result[i].isdigit() and result[i] in letter_fixes:
                result[i] = letter_fixes[result[i]]

        # 2,3,4 positions should be numeric
        for i in [2, 3, 4]:
            if result[i].isalpha() and result[i] in number_fixes:
                result[i] = number_fixes[result[i]]

        # 5,6 positions should be letters
        for i in [5, 6]:
            if result[i].isdigit() and result[i] in letter_fixes:
                result[i] = letter_fixes[result[i]]

        return ''.join(result)

    return plate_text

def ocr_plate(reader, img_bgr, debug=False):
    img_bgr = crop_white_area(img_bgr)
    h, w = img_bgr.shape[:2]

    # Upscale if necessary
    target_h = 400
    if h < target_h:
        scale = target_h / h
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LANCZOS4)
        if debug:
            print(f"Upscaled to: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # Sharpening
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    img_bgr = cv2.filter2D(img_bgr, -1, kernel_sharpen)

    results = reader.readtext(img_bgr, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    if debug:
        print(f"EasyOCR found {len(results)} text regions:")
        for bbox, text, confidence in results:
            text_clean = re.sub(r"[^A-Z0-9]", "", text.upper())
            print(f"  '{text}' -> '{text_clean}' (confidence: {confidence:.3f})")

    valid_results = [(re.sub(r"[^A-Z0-9]", "", text.upper()), conf)
                     for _, text, conf in results if conf > 0.5]

    if valid_results:
        valid_results.sort(key=lambda x: len(x[0]), reverse=True)
        best = valid_results[0][0]
        best_extracted = extract_plate_pattern(best)
        best_fixed = fix_common_ocr_errors(best_extracted)

        if debug:
            print(f"Best OCR: '{best}' -> extracted: '{best_extracted}' -> fixed: '{best_fixed}'")

        return best_fixed

    return ""

def load_onnx_model(model_path: str):
    import onnxruntime as ort
    print(f"onnxruntime version: {ort.__version__}")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    return sess, inp, out

def yolo_onnx_detect(sess, inp_name, out_name, img_bgr, conf=0.35, debug=False):
    h0, w0 = img_bgr.shape[:2]
    size = 640

    # Preprocess
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]

    # Inference
    pred = sess.run([out_name], {inp_name: img})[0]
    pred = pred.squeeze()
    pred = pred.T

    boxes = []

    for detection in pred:
        cx, cy, w, h, confidence = detection
        confidence = float(confidence)

        if confidence > 1:
            confidence = confidence / 640.0

        if confidence < conf:
            continue

        x1 = int((cx - w/2) / size * w0)
        y1 = int((cy - h/2) / size * h0)
        x2 = int((cx + w/2) / size * w0)
        y2 = int((cy + h/2) / size * h0)

        # Clamp
        x1 = max(0, min(x1, w0))
        y1 = max(0, min(y1, h0))
        x2 = max(0, min(x2, w0))
        y2 = max(0, min(y2, h0))

        boxes.append((x1, y1, x2, y2, confidence))

    if debug:
        print(f"DEBUG - YOLO found {len(boxes)} valid boxes above threshold {conf}")

    return boxes

def apply_roi(img_bgr, roi):
    # roi: [x, y, w, h] in relative floats
    h, w = img_bgr.shape[:2]
    rx, ry, rw, rh = roi
    x1 = max(0, int(rx * w))
    y1 = max(0, int(ry * h))
    x2 = min(w, int((rx + rw) * w))
    y2 = min(h, int((ry + rh) * h))
    return img_bgr[y1:y2, x1:x2]

def main():
    opt = get_options()

    roi_raw = opt.get("roi", "0.0,0.0,1.0,1.0")
    if isinstance(roi_raw, str):
        roi = [float(x.strip()) for x in roi_raw.split(",")]
    else:
        roi = roi_raw

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

    # Creates model dir if not exists
    model_dir = os.path.dirname(model_path)
    try:
        os.makedirs(model_dir, exist_ok=True)
        print(f"✅ Model directory ready: {model_dir}")
    except Exception as e:
        print(f"❌ ERROR: Cannot create directory {model_dir}: {e}")
        return

    # check if model exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: ONNX model not found at: {model_path}")
        print(f"")
        print(f"Please download a YOLOv11 license plate detection model and place it at:")
        print(f"  {model_path}")
        print(f"")
        print(f"Recommended model:")
        print(f"  https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/best.onnx")
        print(f"")
        print(f"Instructions:")
        print(f"  1. Create folder: /config/www/smart_gate/ or it will be created for you")
        print(f"  2. Download and rename model to: model.onnx")
        print(f"  3. Upload to the folder above")
        print(f"  4. Restart this addon")
        return

    # check on model size if too small
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    if file_size < 10:
        print(f"⚠️  WARNING: Model file is only {file_size:.1f}MB")
        print(f"   This seems too small for a YOLO model (expected ~200MB)")
        print(f"   The file might be corrupted or a Git LFS pointer.")
        print(f"   Please re-download the model.")
        return

    print(f"✅ Model found: {model_path} ({file_size:.1f}MB)")

    print("Loading YOLO model...")
    sess, inp, out = load_onnx_model(model_path)

    print("Loading EasyOCR... (first run may take 2-3 minutes)")
    reader = easyocr.Reader(['en'], gpu=False)

    last_open = 0.0
    last_motion = ""

    print("Smart Gate started. Watching motion:", motion_entity)
    if debug:
        print(f"DEBUG mode enabled")
        print(f"DEBUG - Confidence threshold: {conf}")
        print(f"DEBUG - ROI: {roi}")
        print(f"DEBUG - Allowed plates: {allowed}")

    while True:
        try:
            state = ha_get_state(motion_entity)
            if state != last_motion:
                print(f"State changed: {last_motion} -> {state}")
                last_motion = state

            if state != "on":
                time.sleep(0.5)
                continue

            now = time.time()
            if now - last_open < cooldown:
                if debug:
                    print(f"In cooldown, skipping ({now - last_open:.1f}s / {cooldown}s)")
                time.sleep(0.5)
                continue

            print("Motion detected! Capturing snapshot...")

            # 1) Snapshot
            ensure_dir(snapshot_path)
            camera_snapshot(camera_entity, snapshot_path)

            # 2) Load snapshot
            img = cv2.imread(snapshot_path)
            if img is None:
                time.sleep(1)
                continue

            img_roi = apply_roi(img, roi)

            if debug:
                print(f"DEBUG - Original image size: {img.shape[1]}x{img.shape[0]}")
                print(f"DEBUG - ROI image size: {img_roi.shape[1]}x{img_roi.shape[0]}")

            # 3) YOLO detect plate
            print(f"Running YOLO detection (confidence threshold: {conf})")
            boxes = yolo_onnx_detect(sess, inp, out, img_roi, conf=conf, debug=debug)
            print(f"YOLO found {len(boxes)} boxes")

            if debug:
                for i, (x1, y1, x2, y2, s) in enumerate(boxes[:3]):
                    print(f"  Box {i}: score={s:.2f}, size={x2-x1}x{y2-y1}, pos=({x1},{y1})")

            if not boxes:
                print("No plates detected")
                time.sleep(1)
                continue

            boxes.sort(key=lambda b: b[4], reverse=True)
            x1, y1, x2, y2, score = boxes[0]
            plate_crop = img_roi[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]

            if plate_crop.size == 0:
                time.sleep(1)
                continue

            plate_crop = remove_plate_border(plate_crop)
            ir_detected, saturation = is_infrared(plate_crop)

            if ir_detected:
                plate_crop = fix_overexposure(plate_crop)
                if debug:
                    print(f"IR mode detected (saturation: {saturation:.1f}) - applying exposure fix")
            else:
                if debug:
                    print(f"Normal mode (saturation: {saturation:.1f}) - skipping exposure fix")

            if debug:
                print(f"Selected box: size={plate_crop.shape[1]}x{plate_crop.shape[0]}")
                ensure_dir(debug_crop_path)
                cv2.imwrite(debug_crop_path, plate_crop)
                print(f"Plate crop saved to {debug_crop_path}")

            # 4) OCR
            plate = ocr_plate(reader, plate_crop, debug=debug)
            print("Detected:", plate, "score:", score)

            # 5) Save history if enabled
            if keep_history:
                os.makedirs(history_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(history_dir, f"{ts}_{plate or 'NONE'}.jpg"), img)

            # 6) Match whitelist -> open gate
            if plate and plate in allowed:
                switch_on(gate_switch)
                last_open = now
                print("Gate opened for:", plate)
            elif debug and plate:
                print(f"DEBUG - Plate '{plate}' not in allowed list")

            # Debounce
            time.sleep(2)

        except Exception as e:
            print("ERROR:", repr(e))
            if debug:
                import traceback
                traceback.print_exc()
            time.sleep(2)

if __name__ == "__main__":
    main()
