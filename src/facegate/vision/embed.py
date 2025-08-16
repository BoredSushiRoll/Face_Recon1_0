from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple

class SFaceEmbedder:
    """
    Uses OpenCV's SFace (cv2.FaceRecognizerSF) to extract embeddings and compare.
    """
    def __init__(self, onnx_path: Path):
        onnx_path = str(Path(onnx_path))
        # backend/target 0 = default (CPU)
        self.model = cv2.FaceRecognizerSF_create(onnx_path, "", 0, 0)

    @staticmethod
    def crop_with_margin(frame_bgr: np.ndarray, box: Tuple[int,int,int,int], margin: float=0.25) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = box
        cx = x + bw/2
        cy = y + bh/2
        size = max(bw, bh) * (1 + margin*2)
        x1 = int(max(0, cx - size/2))
        y1 = int(max(0, cy - size/2))
        x2 = int(min(w, cx + size/2))
        y2 = int(min(h, cy + size/2))
        return frame_bgr[y1:y2, x1:x2].copy()

    def embed_from_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        # SFace expects aligned/cropped face; it handles internal preprocessing.
        # The API: feature = model.feature(face)
        feature = self.model.feature(face_bgr)
        # ensure float32, 1-D
        emb = np.array(feature, dtype=np.float32).reshape(-1)
        # L2 normalize (safety)
        n = np.linalg.norm(emb) + 1e-9
        return emb / n

    def match(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        # cosine sim
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10))
