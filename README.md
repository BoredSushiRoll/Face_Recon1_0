# Face_Recon1_0# FaceGate v1.0 (single webcam)

**Goal:** Real-time face detection + recognition with a clean UI banner:
- Green “Authorized” if the detected face matches **Rareș**.
- Red “Unauthorized” otherwise.

## Stack
- Python 3.11
- OpenCV (capture/draw), onnxruntime (models), PySide6 (UI), pydantic (config)

## Runbook (v1.0 milestones)
1. Enroll Rareș: capture 10–20 images → compute mean embedding → store in SQLite.
2. Live loop: capture → detect faces → embed → compare → paint boxes → banner.
3. Threshold calibration: tune for low false-accept with tolerable false-reject.

## Layout
See `src/facegate/*` modules. Configs in `configs/`. Models in `data/models/`.

## Models (to add manually)
- Detector (e.g., SCRFD) → `data/models/scrfd_500m.onnx`
- Embedding (e.g., ArcFace r100) → `data/models/arcface_r100.onnx`

## Legal/Privacy
Local-only, no cloud, embeddings at rest in SQLite.
