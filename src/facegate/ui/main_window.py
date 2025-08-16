import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget

from facegate.ui.video_widget import VideoWidget
from facegate.vision.detect import detect_faces_haar
from facegate.utils.draw import draw_box_with_label
from facegate.identities.store import IdentityStore
from facegate.vision.embed import SFaceEmbedder
from facegate.vision.match import match_face  # still fine, cosine

class MainWindow(QMainWindow):
    def __init__(self, cap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FaceGate v1.0")

        self.banner = QLabel("UNAUTHORIZED")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet("font-size: 32px; color: red; font-weight: bold;")

        store = IdentityStore("./data/embeddings/identities.db")
        self.known = store.get_all()
        store.close()

        self.embedder = SFaceEmbedder("./data/models/face_recognition_sface_2021dec.onnx")

        def processor(frame_bgr):
            boxes = detect_faces_haar(frame_bgr)
            authorized = False
            for (x,y,w,h) in boxes:
                crop = frame_bgr[y:y+h, x:x+w].copy()
                if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 40:
                    continue
                emb = self.embedder.embed_from_crop(crop)
                best_name, score, is_match = match_face(emb, self.known, threshold=0.5)

                if is_match:
                    authorized = True
                    label = "Rares"  # OpenCV still can’t render ș
                    draw_box_with_label(frame_bgr, (x,y,w,h), label, color=(0, 255, 0))
                else:
                    draw_box_with_label(frame_bgr, (x,y,w,h), "Unknown", color=(0, 0, 255))

            if authorized:
                self.banner.setText("AUTHORIZED")
                self.banner.setStyleSheet("font-size: 32px; color: green; font-weight: bold;")
            else:
                self.banner.setText("UNAUTHORIZED")
                self.banner.setStyleSheet("font-size: 32px; color: red; font-weight: bold;")

            return frame_bgr

        self.video = VideoWidget(cap, processor=processor)

        layout = QVBoxLayout()
        layout.addWidget(self.banner)
        layout.addWidget(self.video)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
