import cv2
import numpy as np
from pathlib import Path
from facegate.identities.store import IdentityStore
from facegate.vision.detect import detect_faces_haar
from facegate.vision.embed import SFaceEmbedder

def main():
    db_path = Path("./data/embeddings/identities.db")
    model_path = Path("./data/models/face_recognition_sface_2021dec.onnx")
    store = IdentityStore(db_path)
    embedder = SFaceEmbedder(model_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera open failed.")
        return

    print("Enrollment: press SPACE to capture a sample when your face is centered/frontal.")
    print("Press ENTER to finish and save; ESC to abort.")
    samples = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        faces = detect_faces_haar(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Enroll Rareș - press SPACE to sample", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:   # ESC
            samples = []
            break
        elif k == 13: # ENTER
            break
        elif k == 32: # SPACE
            if faces:
                # take first detected face
                x,y,w,h = faces[0]
                crop = frame[y:y+h, x:x+w].copy()
                emb = embedder.embed_from_crop(crop)
                samples.append(emb)
                print(f"Captured sample #{len(samples)} (emb dim {emb.shape[0]})")
            else:
                print("No face detected; try again.")

    cap.release()
    cv2.destroyAllWindows()

    if not samples:
        print("No samples captured. Aborting.")
        return

    mean_emb = np.mean(np.stack(samples, axis=0), axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)

    store.add_person("Rareș", mean_emb)
    print(f"Saved Rareș with {len(samples)} samples.")
    store.close()

if __name__ == "__main__":
    main()
