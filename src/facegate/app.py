import os
import sys
import cv2
from PySide6.QtWidgets import QApplication

from facegate.core.config import load_config
from facegate.core.log import setup_logger
from facegate.ui.main_window import MainWindow

def main():
    cfg_path = os.getenv("FACEGATE_CONFIG", "./configs/app.default.yaml")
    cfg = load_config(cfg_path)

    logger = setup_logger(level=cfg.app.log_level)
    logger.info(f"Loaded config from {cfg_path}")
    logger.info(f"App name: {cfg.app.name}, Mode: {cfg.app.mode}")

    cap = cv2.VideoCapture(cfg.camera.index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error(f"Failed to open camera index {cfg.camera.index}")
        return

    app = QApplication(sys.argv)
    window = MainWindow(cap)
    window.resize(cfg.camera.width, cfg.camera.height + 100)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
