import threading
import numpy as np
import cv2

from utils import get_options

# ── Zone state ────────────────────────────────────────────────────────────────
_zones_lock = threading.Lock()
_privacy_zones = []

# ── Motion ref frame ──────────────────────────────────────────────────────────
_motion_ref_frame = None
_motion_ref_lock = threading.Lock()


def _load_privacy_zones():
    """Read privacy_zones from addon options (same source as allowed_plates)."""
    global _privacy_zones
    try:
        opt = get_options()
        zones = opt.get("privacy_zones", [])
        with _zones_lock:
            _privacy_zones = zones
        print(f"[SmartGate] Privacy zones loaded: {len(zones)} zone(s)")
    except Exception as e:
        print(f"[SmartGate] Error loading privacy zones: {e}")


def apply_privacy_mask(frame):
    """
    Black-out all privacy zones on a frame before recognition or notification.
    Thread-safe — reads the zone list atomically via _zones_lock.
    """
    with _zones_lock:
        zones = list(_privacy_zones)
    for zone in zones:
        try:
            cv2.rectangle(
                frame,
                (int(zone["x1"]), int(zone["y1"])),
                (int(zone["x2"]), int(zone["y2"])),
                (0, 0, 0),
                -1,
            )
        except (KeyError, TypeError) as e:
            print(f"[SmartGate] Invalid zone entry {zone}: {e}")
    return frame


def _build_exclusion_mask(frame_shape: tuple) -> np.ndarray:
    """
    Return a uint8 mask (same H×W as frame) where:
      255 = area to CHECK for motion  (outside privacy zones)
        0 = area to IGNORE            (inside privacy zones)
    """
    h, w = frame_shape[:2]
    mask = np.full((h, w), 255, dtype="uint8")
    with _zones_lock:
        zones = list(_privacy_zones)
    for zone in zones:
        try:
            mask[
                int(zone["y1"]) : int(zone["y2"]), int(zone["x1"]) : int(zone["x2"])
            ] = 0
        except (KeyError, TypeError):
            pass
    return mask


def validate_motion_outside_zones(frame, threshold: float = 0.002) -> bool:
    """
    Compare `frame` against the stored reference frame.
    Returns True  → real motion detected outside masked zones → proceed
    Returns False → no significant motion outside zones → skip trigger

    `threshold` = minimum fraction of unmasked pixels that must differ
                  to be considered real motion (default 0.2 %).
    """
    global _motion_ref_frame

    if frame is None:
        return True  # can't validate, let it through

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    with _motion_ref_lock:
        ref = _motion_ref_frame

    if ref is None:
        with _motion_ref_lock:
            _motion_ref_frame = gray
        return True

    if ref.shape != gray.shape:
        with _motion_ref_lock:
            _motion_ref_frame = gray
        return True

    exclusion_mask = _build_exclusion_mask(frame.shape)

    diff = cv2.absdiff(ref, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, thresh, mask=exclusion_mask)

    unmasked_pixels = int(np.count_nonzero(exclusion_mask))
    motion_pixels = int(np.count_nonzero(thresh))

    if unmasked_pixels == 0:
        return False  # entire frame is masked — no valid area to check, skip trigger

    ratio = motion_pixels / unmasked_pixels

    with _motion_ref_lock:
        _motion_ref_frame = gray

    return ratio >= threshold
