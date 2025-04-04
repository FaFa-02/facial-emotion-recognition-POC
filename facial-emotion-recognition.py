import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# def model path
model_path = 'C:\\Users\\fabot\\Downloads\\face_landmarker.task'
# Read Images
mp_image = mp.Image.create_from_file('smiling-woman.jpg')
cv_image = cv2.imread('smiling-woman.jpg')

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
    face_landmarker_result = landmarker.detect(mp_image)

    for landmark in face_landmarker_result.face_landmarks[0]:
        print(landmark)
