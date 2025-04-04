import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# def model path
model_path = 'C:\\Users\\fabot\\Downloads\\face_landmarker.task'

# Read Images
image = mp.Image.create_from_file('smiling-woman.jpg')

# Set configuration options
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

# Initialise landmarker
with FaceLandmarker.create_from_options(options) as landmarker:
    face_landmarker_result = landmarker.detect(image)
