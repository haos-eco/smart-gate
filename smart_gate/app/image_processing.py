import cv2
import numpy as np
import os
import urllib.request

if os.path.exists("/config"):
    # Home Assistant addon (production)
    _SR_MODEL_PATH = "/config/www/smart_gate/models/ai/EDSR_x2.pb"
else:
    # Local development/testing
    _SR_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "ai", "EDSR_x2.pb"
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
            "type": "edsr",
        },
        {
            "name": "FSRCNN_x2.pb",
            "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
            "type": "fsrcnn",
        },
    ]

    for model_info in models:
        try:
            print(f"   Trying {model_info['name']}...")

            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r   Progress: {percent:.1f}%", end="", flush=True)

            urllib.request.urlretrieve(
                model_info["url"], _SR_MODEL_PATH, reporthook=show_progress
            )
            print()

            size_mb = os.path.getsize(_SR_MODEL_PATH) / (1024 * 1024)
            if size_mb < 0.1:
                print(f"   ❌ File too small ({size_mb:.2f}MB), trying next...")
                os.remove(_SR_MODEL_PATH)
                continue

            print(f"   ✅ Downloaded successfully ({size_mb:.2f}MB)")
            model_type_file = _SR_MODEL_PATH.replace(".pb", ".type")
            with open(model_type_file, "w") as f:
                f.write(model_info["type"])
            return True

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            if os.path.exists(_SR_MODEL_PATH):
                os.remove(_SR_MODEL_PATH)
            continue

    print(f"⚠️  Could not download AI SR model — will use bicubic fallback")
    return False


def get_model_type():
    model_type_file = _SR_MODEL_PATH.replace(".pb", ".type")
    if os.path.exists(model_type_file):
        with open(model_type_file, "r") as f:
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
        sr.setModel(get_model_type(), 2)
        _SR_MODEL = sr
        return _SR_MODEL
    except Exception as e:
        print(f"⚠️  Failed to load SR model: {e}")
        return None


def apply_roi(img_bgr, roi):
    """Apply region of interest to image.
    Args:
        img_bgr: BGR image
        roi: [x1, y1, x2, y2] in relative floats (0.0 to 1.0)
    """
    h, w = img_bgr.shape[:2]
    rx, ry, rw, rh = roi
    x1 = max(0, int(rx * w))
    y1 = max(0, int(ry * h))
    x2 = min(w, int((rx + rw) * w))
    y2 = min(h, int((ry + rh) * h))
    return img_bgr[y1:y2, x1:x2]


def remove_plate_border(img_crop):
    """Remove plate frame borders"""
    h, w = img_crop.shape[:2]
    margin_h = int(h * 0.18)
    margin_w = int(w * 0.08)
    return img_crop[margin_h : h - margin_h, margin_w : w - margin_w]


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


def is_overexposed(img_bgr, threshold: float = 200.0) -> bool:
    """Detect if plate crop is overexposed by headlights.

    When headlights hit the plate directly, most pixels saturate to ~255
    making characters invisible to OCR.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > threshold


def fix_headlight_overexposure(img_bgr, debug: bool = False) -> np.ndarray:
    """Recover plate characters from headlight overexposure via gamma + CLAHE + invert."""
    # Gamma correction — compresses saturated highlights back into readable range
    gamma = 0.4
    lut = np.array(
        [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)], dtype=np.uint8
    )
    img_bgr = cv2.LUT(img_bgr, lut)

    if debug:
        mean = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        print(f"  Headlight fix: gamma={gamma}, mean after: {mean:.1f}")

    # CLAHE — local contrast recovery after gamma
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Invert if background is still predominantly light
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:
        img_bgr = cv2.bitwise_not(img_bgr)
        if debug:
            print("  Headlight fix: inverted (light background)")

    return img_bgr


def deskew_plate(img):
    """Correct plate skew using Hough lines. Angles <0.5° are ignored."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)

    if lines is None:
        return img

    angles = []
    for line in lines[:15]:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) < 25:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img

    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def apply_sr_and_upscale(img_bgr, debug=False):
    """SR 2x + upscale to minimum 400px height."""
    h, w = img_bgr.shape[:2]

    sr = get_sr_model()
    if sr is not None:
        try:
            img_bgr = sr.upsample(img_bgr)
            if debug:
                print(f"  SR 2x: {w}x{h} → {img_bgr.shape[1]}x{img_bgr.shape[0]}")
        except Exception as e:
            if debug:
                print(f"  SR error: {e}, using bicubic fallback")
            img_bgr = cv2.resize(
                img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
            )
    else:
        img_bgr = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        if debug:
            print(f"  Bicubic 2x: {w}x{h} → {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    h_new, w_new = img_bgr.shape[:2]
    if h_new < 400:
        scale = 400 / h_new
        img_bgr = cv2.resize(
            img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
        )
        if debug:
            print(
                f"  LANCZOS4: {w_new}x{h_new} → {img_bgr.shape[1]}x{img_bgr.shape[0]}"
            )

    return img_bgr


def preprocess_plate(img_bgr, debug=False):
    """
    Full preprocessing pipeline for a plate crop before OCR.

    Pipeline:
      1. crop_white_area     — remove EU blue strip
      2. fix_headlight_overexposure — gamma + CLAHE + invert (only if overexposed)
      3. bilateral denoise   — reduce noise while preserving character edges
      4. CLAHE               — improve local contrast (always applied)
      5. unsharp mask        — sharpen character edges
      6. deskew              — correct plate skew
      7. SR + upscale        — 2x super-resolution, minimum 400px height
    """
    # 1. Remove EU blue strip
    img_bgr = crop_white_area(img_bgr)
    if debug:
        print("  Preprocess: EU strip cropped")

    # 2. Headlight overexposure — must come before CLAHE
    if is_overexposed(img_bgr):
        if debug:
            print("  Preprocess: overexposure detected — applying headlight fix")
        img_bgr = fix_headlight_overexposure(img_bgr, debug=debug)

    # 3. Bilateral denoise
    img_bgr = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    if debug:
        print("  Preprocess: bilateral denoise applied")

    # 4. CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if debug:
        print("  Preprocess: CLAHE applied")

    # 5. Unsharp mask
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.5)
    img_bgr = cv2.addWeighted(img_bgr, 1.8, gaussian, -0.8, 0)
    if debug:
        print("  Preprocess: unsharp mask applied")

    # 6. Deskew
    img_bgr = deskew_plate(img_bgr)
    if debug:
        print("  Preprocess: deskew applied")

    # 7. SR + upscale
    img_bgr = apply_sr_and_upscale(img_bgr, debug=debug)

    return img_bgr
