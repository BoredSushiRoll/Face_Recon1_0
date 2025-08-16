import cv2
from typing import Optional, Callable
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

# Callable signature: (frame_bgr) -> frame_bgr
FrameProcessor = Callable[[any], any]

class VideoWidget(QLabel):
    def __init__(self, cap, processor: Optional[FrameProcessor] = None, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.processor = processor
        self.setAlignment(Qt.AlignCenter)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        if self.processor is not None:
            frame = self.processor(frame)

        # BGR -> RGB for Qt
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img))
