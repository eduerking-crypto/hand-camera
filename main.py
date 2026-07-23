import cv2
import mediapipe as mp
import numpy as np

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hands_data = []

def result_callback(result, image, timestamp):
    global hands_data
    hands_data = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            hands_data.append(hand)

def classify_fingers(lms):
    fingers = []
    for tip, pip, thumb in [(4, 3, True), (8, 6, False), (12, 10, False), (16, 14, False), (20, 18, False)]:
        if thumb:
            fingers.append(lms[tip].x < lms[pip].x)
        else:
            fingers.append(lms[tip].y < lms[pip].y)
    return sum(fingers)

cap = cv2.VideoCapture(0)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=2,
    min_hand_detection_confidence=0.6
)

landmarker = HandLandmarker.create_from_options(options)
ts = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts += 1
    landmarker.detect_async(mp_image, ts)

    for hand_lms in hands_data:
        for i, lm in enumerate(hand_lms):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        count = classify_fingers(hand_lms)
        cv2.putText(frame, f"Dedos: {count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
