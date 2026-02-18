import cv2
import numpy as np

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
