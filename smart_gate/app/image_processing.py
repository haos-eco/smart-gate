import cv2
import numpy as np
import os
import urllib.request

if os.path.exists('/config'):
    # Home Assistant addon (production)
    _SR_MODEL_PATH = "/config/www/smart_gate/models/ai/EDSR_x2.pb"
else:
    # Local development/testing
    _SR_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "ai",
        "EDSR_x2.pb"
    )

_SR_MODEL = None

def download_sr_model():
    """Download AI super-resolution model if not present"""
    model_dir = os.path.dirname(_SR_MODEL_PATH)

    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError as e:
        print(f"⚠️  Cannot create model directory {model_dir}: {e}")
        return False

    if os.path.exists(_SR_MODEL_PATH):
        return True

    print(f"📥 Downloading AI super-resolution model...")
    print(f"   Destination: {_SR_MODEL_PATH}")

    models = [
        {
            "name": "EDSR_x2.pb",
            "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
            "type": "edsr"
        },
        {
            "name": "FSRCNN_x2.pb",
            "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
            "type": "fsrcnn"
        }
    ]

    for model_info in models:
        try:
            print(f"   Trying {model_info['name']}...")

            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r   Progress: {percent:.1f}%", end='', flush=True)

            urllib.request.urlretrieve(
                model_info['url'],
                _SR_MODEL_PATH,
                reporthook=show_progress
            )
            print()

            size_mb = os.path.getsize(_SR_MODEL_PATH) / (1024 * 1024)
            if size_mb < 0.1:
                print(f"   ❌ File too small ({size_mb:.2f}MB), trying next...")
                os.remove(_SR_MODEL_PATH)
                continue

            print(f"   ✅ Downloaded successfully ({size_mb:.2f}MB)")

            model_type_file = _SR_MODEL_PATH.replace('.pb', '.type')
            with open(model_type_file, 'w') as f:
                f.write(model_info['type'])

            return True

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            if os.path.exists(_SR_MODEL_PATH):
                os.remove(_SR_MODEL_PATH)
            continue

    print(f"⚠️  Could not download AI SR model from any source")
    print(f"   Will use bicubic fallback")
    return False

def get_model_type():
    """Get the type of downloaded model"""
    model_type_file = _SR_MODEL_PATH.replace('.pb', '.type')
    if os.path.exists(model_type_file):
        with open(model_type_file, 'r') as f:
            return f.read().strip()
    return "edsr"

def get_sr_model():
    """Get or initialize super-resolution model"""
    global _SR_MODEL

    if _SR_MODEL is not None:
        return _SR_MODEL

    if not download_sr_model():
        return None

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(_SR_MODEL_PATH)

        model_type = get_model_type()
        sr.setModel(model_type, 2)  # 2x scale

        _SR_MODEL = sr
        return _SR_MODEL

    except Exception as e:
        print(f"⚠️  Failed to load SR model: {e}")
        print(f"   Will use bicubic fallback")
        return None

def apply_roi(img_bgr, roi):
    """Apply region of interest to image

    Args:
        img_bgr: BGR image
        roi: [x, y, w, h] in relative floats (0.0 to 1.0)
    """

    h, w = img_bgr.shape[:2]
    rx, ry, rw, rh = roi
    x1 = max(0, int(rx * w))
    y1 = max(0, int(ry * h))
    x2 = min(w, int((rx + rw) * w))
    y2 = min(h, int((ry + rh) * h))
    return img_bgr[y1:y2, x1:x2]

def is_infrared(img_bgr):
    """Detect if image is in infrared mode (low saturation)"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    is_ir = mean_saturation < 20
    return is_ir, mean_saturation

def remove_plate_border(img_crop):
    """Remove plate frame borders"""
    h, w = img_crop.shape[:2]
    margin_h = int(h * 0.18)
    margin_w = int(w * 0.08)
    return img_crop[margin_h:h-margin_h, margin_w:w-margin_w]

def fix_overexposure(img_bgr):
    """Fix overexposure using CLAHE"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def crop_white_area(img_bgr):
    """Crop only white area of plate (removes EU blue strip)"""
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

def enhance_plate_ai_sr(img_bgr, debug=False):
    # AI super-resolution with minimal post-processing

    h, w = img_bgr.shape[:2]
    sr = get_sr_model()

    if sr is not None:
        try:
            img_bgr = sr.upsample(img_bgr)
            if debug:
                print(f"AI SR 2x: {w}x{h} → {img_bgr.shape[1]}x{img_bgr.shape[0]}")
        except Exception as e:
            if debug:
                print(f"AI SR error: {e}, using bicubic")
            img_bgr = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    else:
        img_bgr = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        if debug:
            print(f"Bicubic 2x: {w}x{h} → {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # Upscale to 400px height (if needed)
    h_new, w_new = img_bgr.shape[:2]
    if h_new < 400:
        scale = 400 / h_new
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        if debug:
            print(f"LANCZOS4: {w_new}x{h_new} → {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # Light sharpening
    kernel_sharpen = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    img_bgr = cv2.filter2D(img_bgr, -1, kernel_sharpen)
    if debug:
        print("Light sharpening applied")

    return img_bgr
