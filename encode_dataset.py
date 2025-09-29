import os
import pickle
import cv2
from deepface import DeepFace

DATASET_DIR = "dataset"       # folder with student images
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_NAME = "Facenet"        # using Facenet for better accuracy
FRAME_SIZE = (320, 240)       # Resize all images before encoding

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
                # Load + resize image for consistency
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARNING] Could not read {img_path}")
                    continue
                img = cv2.resize(img, FRAME_SIZE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                reps = DeepFace.represent(
                    img_path=img_rgb,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )
                if reps and isinstance(reps, list):
                    embeddings[student_name].append(reps[0]["embedding"])
            except Exception as e:
                print(f"[WARNING] Failed on {img_path}: {e}")

    # Save all embeddings
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"[DONE] Embeddings saved to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    encode_dataset()
