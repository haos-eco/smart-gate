import cv2
import numpy as np

def load_model(model_path: str):
    """Load YOLO ONNX model"""
    import onnxruntime as ort
    print(f"onnxruntime version: {ort.__version__}")

    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name

    return sess, inp, out

def detect_plates(sess, inp_name, out_name, img_bgr, conf=0.35, debug=False):
    """Detect license plates using YOLO"""
    h0, w0 = img_bgr.shape[:2]
    size = 640

    # Preprocess
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]

    # Inference
    pred = sess.run([out_name], {inp_name: img})[0]
    pred = pred.squeeze().T

    boxes = []

    for detection in pred:
        cx, cy, w, h, confidence = detection
        confidence = float(confidence)

        if confidence > 1:
            confidence /= 640.0

        if confidence < conf:
            continue

        # Convert from center format to corner format
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
