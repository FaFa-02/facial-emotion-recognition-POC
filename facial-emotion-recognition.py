import cv2
import mediapipe as mp
import numpy as np
import sys
from mediapipe import solutions
from mediapipe.python.solutions import drawing_utils
from mediapipe.framework.formats import landmark_pb2

# def model path
model_path = 'C:\\Users\\fabot\\Downloads\\face_landmarker.task'

"""
# Read Images
mp_image = mp.Image.create_from_file('smiling-woman.jpg')
cv_image = cv2.imread('smiling-woman.jpg')
"""

# Set configuration options
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Draws landmarks ontop of the image
def draw_landmarks_on_image(rgb_image, detection_result):
  """ Code used from https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb """

  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

cap = cv2.VideoCapture(0)

# Initialise landmarker
with FaceLandmarker.create_from_options(options) as landmarker:
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        #results = landmarker.detect(mp_image)
        results = landmarker.detect_async(mp_image, landmarker._result_callback)

        # Annotated version of original image
        annotated_image = draw_landmarks_on_image(mp_image, results)

        cv2.imshow('Video capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cv2.imshow('Image', annotated_image)
    #cv2.waitKey(0)
