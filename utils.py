import cv2
import pickle
import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet"
THRESHOLD = 0.6   # Recommended for Facenet
FRAME_SKIP = 5    # Process 1 frame out of every 5 (~6 fps if video is 30fps)
FRAME_SIZE = (320, 240)  # Resize frames before embedding

def load_embeddings(path="embeddings.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def recognize_face_embedding(query_vec, embeddings):
    best_match, best_score = "Unknown", -1
    for name, vecs in embeddings.items():
        for ref_vec in vecs:
            score = cosine_similarity(query_vec, ref_vec)
            if score > best_score:
                best_score, best_match = score, name
    return best_match if best_score >= THRESHOLD else "Unknown"

def process_video(video_path, embeddings):
    cap = cv2.VideoCapture(video_path)
    recognized = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue
        frame_count += 1

        # Resize for speed
        frame = cv2.resize(frame, FRAME_SIZE)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            reps = DeepFace.represent(
                img_path=rgb_frame,
                model_name=MODEL_NAME,
                enforce_detection=False
            )
            if reps and isinstance(reps, list):
                query_vec = reps[0]["embedding"]
                name = recognize_face_embedding(query_vec, embeddings)
                if name != "Unknown":
                    recognized.add(name)
        except Exception:
            continue

    cap.release()
    return list(recognized)
