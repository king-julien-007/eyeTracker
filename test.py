import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Get screen size
screen_width, screen_height = pyautogui.size()

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for natural movement
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for face landmarks
    results = face_mesh.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmarks (approximate left and right eye indices)
            left_eye = face_landmarks.landmark[474]  # Example: Index 474
            right_eye = face_landmarks.landmark[469] # Example: Index 469
            
            # Convert normalized coords to screen coords
            eye_x = int(left_eye.x * frame_width)
            eye_y = int(left_eye.y * frame_height)

            # Map eye movement to screen movement
            screen_x = np.interp(eye_x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(eye_y, [0, frame_height], [0, screen_height])

            # Move mouse cursor
            pyautogui.moveTo(screen_x, screen_y)

            # Draw eyes for visualization
            cv2.circle(frame, (eye_x, eye_y), 5, (0, 255, 0), -1)

    cv2.imshow("Eye Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
