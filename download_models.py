import urllib.request
import os

models = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

for name, url in models.items():
    if not os.path.exists(name):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, name)
        print(f"Done: {name}")
    else:
        print(f"{name} already exists")
