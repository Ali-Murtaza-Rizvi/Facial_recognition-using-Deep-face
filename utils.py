import cv2
import pickle
import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet"   # You can try "Facenet512" for stronger embeddings
THRESHOLD = 0.55         # Adjust after testing

# -------------------- Helpers --------------------

def load_embeddings(path="embeddings.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def normalize(vec):
    """L2 normalize an embedding vector"""
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)

def cosine_similarity(vec1, vec2):
    """Cosine similarity between two vectors"""
    vec1, vec2 = normalize(vec1), normalize(vec2)
    return np.dot(vec1, vec2)

# -------------------- Recognition --------------------

def recognize_face(face_img, embeddings):
    try:
        reps = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            enforce_detection=False
        )
        if not reps or not isinstance(reps, list):
            return "Unknown"

        query_vec = reps[0]["embedding"]
        query_vec = normalize(query_vec)

        best_match, best_score = "Unknown", -1

        for name, vecs in embeddings.items():
            for ref_vec in vecs:
                ref_vec = normalize(ref_vec)
                score = cosine_similarity(query_vec, ref_vec)
                print(f"[DEBUG] Comparing with {name}: {score:.3f}")
                if score > best_score:
                    best_score, best_match = score, name

        print(f"[DEBUG] Best match: {best_match} ({best_score:.3f})")
        return best_match if best_score >= THRESHOLD else "Unknown"

    except Exception as e:
        print(f"[ERROR] Face recognition failed: {e}")
        return "Unknown"

# -------------------- Video Processing --------------------

def process_video(video_path, embeddings):
    cap = cv2.VideoCapture(video_path)
    recognized = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            reps = DeepFace.represent(
                img_path=rgb_frame,
                model_name=MODEL_NAME,
                enforce_detection=False
            )
            if reps and isinstance(reps, list):
                name = recognize_face(rgb_frame, embeddings)
                if name != "Unknown":
                    recognized.add(name)
        except Exception:
            continue

    cap.release()
    return list(recognized)
