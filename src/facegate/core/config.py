from pathlib import Path
from pydantic import BaseModel
import yaml

class AppMeta(BaseModel):
    name: str = "FaceGate"
    mode: str = "single_camera"
    log_level: str = "INFO"

class CameraConfig(BaseModel):
    index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30

class DetectionConfig(BaseModel):
    min_conf: float = 0.6
    min_face_px: int = 100

class RecognitionConfig(BaseModel):
    embed_model_path: Path
    detector_model_path: Path
    metric: str = "cosine"
    threshold: float = 0.45


class UIConfig(BaseModel):
    banner_policy: str = "any_match"
    show_fps: bool = True

class StorageConfig(BaseModel):
    sqlite_path: Path
    enrollment_dir: Path

class AppConfig(BaseModel):
    app: AppMeta
    camera: CameraConfig
    detection: DetectionConfig
    recognition: RecognitionConfig
    ui: UIConfig
    storage: StorageConfig

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
