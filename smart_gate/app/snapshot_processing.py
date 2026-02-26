import os
import time
import cv2

from constants import STATUS_COLORS


def annotate_snapshot(
    snapshot_path: str,
    plate: str,
    bbox,
    yolo_score: float,
    ocr_conf: float,
    status: str = "unknown",
) -> str:
    """
    Draw bounding box + plate label on a snapshot and save an annotated copy.

    status: "opened" | "rejected" | "unknown" | "unreadable"
    Returns the path of the annotated file, or the original path on failure.
    """
    color = STATUS_COLORS.get(status, (200, 200, 200))

    try:
        frame = cv2.imread(snapshot_path)
        if frame is None:
            return snapshot_path

        h, w = frame.shape[:2]

        # ── bounding box ──
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{plate}  YOLO:{yolo_score:.2f}  OCR:{ocr_conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.45, w / 1800)
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            pad = 4
            lx1, ly1 = x1, max(0, y1 - th - pad * 2)
            lx2, ly2 = x1 + tw + pad * 2, y1
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(
                frame,
                label,
                (lx1 + pad, ly2 - pad),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        # ── bottom status bar ──
        bar_h = max(28, int(h * 0.04))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        bar_label = f"SmartGate  |  {plate}  |  {status.upper()}  |  {time.strftime('%H:%M:%S')}"
        font_scale_bar = max(0.38, w / 2200)
        cv2.putText(
            frame,
            bar_label,
            (8, h - bar_h + int(bar_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_bar,
            color,
            1,
            cv2.LINE_AA,
        )

        annotated_dir = os.path.join(os.path.dirname(snapshot_path), "annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        out_path = os.path.join(annotated_dir, os.path.basename(snapshot_path))
        cv2.imwrite(out_path, frame)
        return out_path

    except Exception as e:
        print(f"⚠️  annotate_snapshot error: {e}")
        return snapshot_path
