import cv2
from facegate.core.config import AppConfig

def run_capture_test(cfg: AppConfig, logger):
    cap = cv2.VideoCapture(cfg.camera.index, cv2.CAP_DSHOW)  # DirectShow on Windows is less flaky
    if not cap.isOpened():
        logger.error(f"Failed to open camera index {cfg.camera.index}")
        return

    # Set properties (best-effort; some cams ignore these)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    cap.set(cv2.CAP_PROP_FPS,          cfg.camera.target_fps)

    logger.info("Camera opened. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.warning("Failed to read frame.")
            break

        cv2.imshow("FaceGate - Camera Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera closed.")
