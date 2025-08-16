import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget

from facegate.ui.video_widget import VideoWidget
from facegate.vision.detect import detect_faces_haar
from facegate.utils.draw import draw_box_with_label
from facegate.identities.store import IdentityStore
from facegate.vision.match import match_face

class MainWindow(QMainWindow):
    def __init__(self, cap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FaceGate v1.0")

        self.banner = QLabel("UNAUTHORIZED")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet("font-size: 32px; color: red; font-weight: bold;")

        # load identities from DB
        store = IdentityStore("./data/embeddings/identities.db")
        self.known = store.get_all()
        store.close()

        def processor(frame_bgr):
            boxes = detect_faces_haar(frame_bgr)
            authorized = False
            for box in boxes:
                # stub embedding: pretend each detection = 512-dim ones
                emb = np.ones(512, dtype=np.float32)
                best_name, score, is_match = match_face(emb, self.known, threshold=0.5)

                if is_match:
                    authorized = True
                    draw_box_with_label(frame_bgr, box, best_name, color=(0, 255, 0))
                else:
                    draw_box_with_label(frame_bgr, box, "Unknown", color=(0, 0, 255))

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
