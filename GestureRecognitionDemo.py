import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Draw hands on video 
mp_drawing = mp.solutions.drawing_utils

# Use default computer camera
cap = cv2.VideoCapture(0)

def count_fingers_up(hand_landmarks, handedness):
    fingers_tip_indices = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                           mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    fingers_pip_indices = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                           mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    
    fingers_up = 0
    fingers_raised = True
    
    # Check fingers raised
    for tip_index, pip_index in zip(fingers_tip_indices, fingers_pip_indices):
        tip = hand_landmarks.landmark[tip_index]
        pip = hand_landmarks.landmark[pip_index]
        
        if tip.y < pip.y:  # if tip is higher than pip joint, count
            fingers_up += 1
        else:
            fingers_raised = False
    
    # To avoid constantly counting the thumb, only count if 4 other fingers are raised
    if fingers_raised:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  # Intermediate phalangeal joint of the thumb
        
        # Logic to check if thumb is raised, could vary based on the hand orientation and specific gesture definition
        # Simple heuristic: for right hand, check if thumb tip is to the left of the thumb IP joint (and vice versa for left hand)
        if handedness == 'Right' and thumb_tip.x < thumb_ip.x:
            fingers_up += 1
        elif handedness == 'Left' and thumb_tip.x > thumb_ip.x:
            fingers_up += 1

    return fingers_up

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Process Image
    # Flip the image horizontally for a later selfie-view display and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, mark the image as not writeable to pass by reference
    image.flags.writeable = False
    results = hands.process(image)
    
    # Draw the hand annotations
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Determine the handedness (left or right hand)
            handedness = handedness_info.classification[0].label
            
            # Count fingers up, including the thumb logic as defined
            fingers_up = count_fingers_up(hand_landmarks, handedness)
            print(f"Number of fingers up ({handedness} hand): {fingers_up}")
            
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting image
    cv2.imshow('MediaPipe Hands', image)
    
    # Break the loop when the user presses 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
