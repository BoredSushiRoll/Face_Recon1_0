import cv2
from typing import Tuple

def draw_box_with_label(
    frame_bgr,
    box: Tuple[int, int, int, int],
    label: str,
    color=(0, 0, 255),  # BGR (red default)
    thickness: int = 2,
):
    x, y, w, h = box
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, thickness)
    # label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(frame_bgr, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_bgr
