import urllib.request
from deepface import DeepFace

# Download a sample image (if not already present)
img_url = "https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg"
img_path = "sample.jpg"

urllib.request.urlretrieve(img_url, img_path)

# Run DeepFace representation
print("Running DeepFace on sample image...")
result = DeepFace.represent(img_path=img_path, enforce_detection=False)

print("\nRepresentation vector:")
print(result)
