import urllib.request, os

url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
name = "hand_landmarker.task"
if not os.path.exists(name):
    print(f"Descargando {name}...")
    urllib.request.urlretrieve(url, name)
    print("OK")
else:
    print("Ya existe")
