import os
import pickle
from deepface import DeepFace

DATASET_DIR = "dataset"       # folder with student images
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_NAME = "Facenet"        # MUST match utils.py

def encode_dataset():
    embeddings = {}

    for student_name in os.listdir(DATASET_DIR):
        student_dir = os.path.join(DATASET_DIR, student_name)
        if not os.path.isdir(student_dir):
            continue

        print(f"[INFO] Encoding images for: {student_name}")
        embeddings[student_name] = []

        for img_file in os.listdir(student_dir):
            img_path = os.path.join(student_dir, img_file)
            try:
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )
                if reps and isinstance(reps, list):
                    embeddings[student_name].append(reps[0]["embedding"])
            except Exception as e:
                print(f"[WARNING] Failed on {img_path}: {e}")

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"[DONE] Embeddings saved to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    encode_dataset()
