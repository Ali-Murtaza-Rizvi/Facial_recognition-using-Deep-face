# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from utils import process_video, load_embeddings

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
DATASET_FOLDER = "dataset"
EMBEDDINGS_PATH = "embeddings.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load precomputed embeddings
embeddings = load_embeddings(EMBEDDINGS_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return "No video uploaded", 400

        video = request.files["video"]
        if video.filename == "":
            return "No file selected", 400

        # Save uploaded video
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        # Process video → recognize faces
        recognized = process_video(video_path, embeddings)

        # Save results to a file
        result_file = os.path.join(RESULTS_FOLDER, "recognized_students.txt")
        with open(result_file, "w") as f:
            f.write("Recognized Students:\n")
            for name in recognized:
                f.write(f"- {name}\n")

        return redirect(url_for("results"))

    return render_template("index.html")


@app.route("/results")
def results():
    result_file = os.path.join(RESULTS_FOLDER, "recognized_students.txt")
    if not os.path.exists(result_file):
        return "No results yet. Upload a video first.", 400

    with open(result_file, "r") as f:
        content = f.read().splitlines()

    return render_template("results.html", content=content)


if __name__ == "__main__":
    app.run(debug=True)
