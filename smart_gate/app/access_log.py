import time
from constants import MAX_ENTRIES
from typing import Optional
from utils import load_logs, save_logs

def log_event(
        plate: str,
        status: str,                    # "opened" | "rejected" | "unknown"
        snapshot_path: str,
        match_type: str,                # "exact" | "fuzzy" | "none"
        yolo_score: float = 0.0,
        ocr_conf: float = 0.0,
        matched_plate: Optional[str] = None,
        reason: Optional[str] = None,   # human-readable reason for rejection
):
    """Append an access event to the log."""
    entries = load_logs()

    # Snapshot served via /local/
    snapshot_url = snapshot_path.replace("/config/www/", "/local/") if snapshot_path else None

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timestamp_unix": int(time.time()),
        "plate": plate,
        "status": status,
        "match_type": match_type,
        "matched_plate": matched_plate,
        "yolo_score": round(yolo_score, 3),
        "ocr_conf": round(ocr_conf, 3),
        "snapshot_url": snapshot_url,
        "reason": reason,
    }

    entries.insert(0, entry)
    entries = entries[:MAX_ENTRIES]
    save_logs(entries)
