#8Unit08_3hand3.py
import cv2
import mediapipe as mp
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
landmarker = vision.HandLandmarker.create_from_options(options)
hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

cap = cv2.VideoCapture(0)
run = True
rx, ry, count = 0, 0, 0
color_box = (0, 0, 255)

while cap.isOpened():
    success, frame = cap.read()
    img = cv2.resize(frame, (640, 420))
    w, h = (img.shape[1], img.shape[0])
    if run:   # Initialize box position
        run = False
        rx = random.randint(10, w - 80)
        ry = random.randint(10, h - 80)
        color_box = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        print("New box:", rx, ry, "Color:", color_box)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgrgb)
    detection_result = landmarker.detect(mp_image)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            pixel_landmarks = []
            for lm in hand_landmarks:
                pixel_landmarks.append((int(lm.x * w), int(lm.y * h)))
            for connection in hand_connections:   # Draw connections
                start_idx = connection.start
                end_idx = connection.end
                cv2.line(img, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (200, 200, 0), 2)
            for point in pixel_landmarks:   # Draw points
                cv2.circle(img, point, 3, (0, 0, 255), -1) # Red points
            if len(pixel_landmarks) > 20:   # Landmark 20 is Pinky Tip
                x = pixel_landmarks[20][0]
                y = pixel_landmarks[20][1]
                if (rx < x < rx + 80) and (ry < y < ry + 80):  # Check collision with Pinky Tip
                    count += 1
                    run = True

    cv2.rectangle(img, (rx, ry), (rx + 80, ry + 80), color_box, 5)
    img = cv2.flip(img, 1)
    cv2.putText(img, 'Score:'+str(count), (30, 80), 2, 2, (0, 0, 255), 2)
    cv2.imshow('Unit08_3 | StudentID | hand3', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
landmarker.close()
cv2.destroyAllWindows()
