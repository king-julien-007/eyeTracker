import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = face_landmarks.landmark[474]
            right_eye = face_landmarks.landmark[469]
            
            eye_x = int(left_eye.x * frame_width)
            eye_y = int(left_eye.y * frame_height)

            screen_x = np.interp(eye_x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(eye_y, [0, frame_height], [0, screen_height])

            pyautogui.moveTo(screen_x, screen_y)

            cv2.circle(frame, (eye_x, eye_y), 5, (0, 255, 0), -1)

    cv2.imshow("Eye Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
