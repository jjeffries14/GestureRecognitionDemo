import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def count_fingers_up(hand_landmarks, handedness):
    # Fingers tip indices (excluding thumb)
    fingers_tip_indices = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                           mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    fingers_pip_indices = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                           mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    
    fingers_up = 0
    all_four_fingers_raised = True
    
    # Check four fingers (excluding thumb)
    for tip_index, pip_index in zip(fingers_tip_indices, fingers_pip_indices):
        tip = hand_landmarks.landmark[tip_index]
        pip = hand_landmarks.landmark[pip_index]
        
        if tip.y < pip.y:  # Finger tip is above the corresponding PIP joint
            fingers_up += 1
        else:
            all_four_fingers_raised = False
    
    # Check thumb if all four fingers are raised
    if all_four_fingers_raised:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        
        # Simple heuristic for thumb position
        if handedness == 'Right' and thumb_tip.x < thumb_ip.x:
            fingers_up += 1
        elif handedness == 'Left' and thumb_tip.x > thumb_ip.x:
            fingers_up += 1

    return fingers_up

# Tkinter GUI setup
window = tk.Tk()
window.title("Gesture Recognition")

camera_status_label = tk.Label(window, text="Initializing camera...")
camera_status_label.pack()

fingers_count_label = tk.Label(window, text="Number of fingers up: ")
fingers_count_label.pack()

video_canvas = tk.Canvas(window, width=640, height=480)
video_canvas.pack()

# Start video capture
cap = cv2.VideoCapture(0)

def update_video_frame():
    ret, frame = cap.read()
    if ret:
        camera_status_label.config(text="Camera is transmitting")
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness = handedness_info.classification[0].label
                fingers_up = count_fingers_up(hand_landmarks, handedness)
                fingers_count_label.config(text=f"Number of fingers up ({handedness} hand): {fingers_up}")
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        video_canvas.create_image(0, 0, anchor=tk.NW, image=frame)
        video_canvas.image = frame
    else:
        camera_status_label.config(text="Camera not transmitting")

    window.after(10, update_video_frame)

update_video_frame()
window.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
