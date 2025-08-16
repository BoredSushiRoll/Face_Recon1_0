import logging
import sys

def setup_logger(name: str = "facegate", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
