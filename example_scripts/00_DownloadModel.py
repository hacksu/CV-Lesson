import urllib.request
from pathlib import Path


models = {"hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"}
    

for filename, url in models.items():
    path = Path(filename)
    if path.exists():
        print(f"Model {filename}")
        continue
    print(f"Downloading {filename} from {url}")
    urllib.request.urlretrieve(url, filename)