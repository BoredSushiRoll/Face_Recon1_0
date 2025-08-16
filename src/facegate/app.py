import os
from facegate.core.config import load_config
from facegate.core.log import setup_logger
from facegate.pipeline.engine import run_capture_test

def main():
    cfg_path = os.getenv("FACEGATE_CONFIG", "./configs/app.default.yaml")
    cfg = load_config(cfg_path)

    logger = setup_logger(level=cfg.app.log_level)
    logger.info(f"Loaded config from {cfg_path}")
    logger.info(f"App name: {cfg.app.name}, Mode: {cfg.app.mode}")
    logger.info(f"Camera index: {cfg.camera.index}, resolution: {cfg.camera.width}x{cfg.camera.height}")

    # temp: run webcam test loop
    run_capture_test(cfg, logger)

if __name__ == "__main__":
    main()
