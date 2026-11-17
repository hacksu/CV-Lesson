import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Define standard 21-point hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

def draw_landmarks(frame, detection_result):
    h, w, c = frame.shape
    for hand_landmarks in detection_result.hand_landmarks:
        # onvert normalized coordinates (0.0 to 1.0) to pixel coordinates
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        #Draw skeleton lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

        #Draw joint dots
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
    return frame

#Configure the HandLandmarker for live video
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#Start webcam capture
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
print("Camera opened:", cap.isOpened())

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #Flip horizontally for a natural mirror view and convert BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        #Pass a monotonically increasing timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        #Draw detections and display
        frame = draw_landmarks(frame, result)
        cv2.imshow("MediaPipe Hand Detection", frame)

        #Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
