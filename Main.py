import depthai as dai
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import mediapipe as mp

class HandGestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.start_gesture_detected_frame_count = 0
        self.frames_to_hold_gesture = 300
        self.new_number_gesture_detected = False
        self.numbers_sum = [0]  
        self.current_sum_index = 0  
        self.current_detected_number = None
        self.consecutive_frame_count = 0
        self.consecutive_frames_required = 20
        self.consecutive_new_number_frames = 0

   
    def detect_start_gesture(self, hand_landmarks):
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        if not (thumb_ip.y < thumb_tip.y * 1.1 and thumb_ip.y > thumb_tip.y * 0.9):
            return False

        for finger_tip, finger_pip in [(self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
                                        (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                                        (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
                                        (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)]:
            tip = hand_landmarks.landmark[finger_tip]
            pip = hand_landmarks.landmark[finger_pip]
            if tip.y <= pip.y:
                return False

        return True

    def detect_new_number_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]

        if thumb_tip.x < index_dip.x:
            for finger_tip, finger_pip in [(self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
                                           (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                                           (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
                                           (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)]:
                tip = hand_landmarks.landmark[finger_tip]
                pip = hand_landmarks.landmark[finger_pip]
                if tip.y < pip.y:
                    return False
            return True
        return False

    def count_fingers(self, hand_landmarks):
        
        raised_fingers = 0

        # Check thumb separately
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]
        if thumb_tip.x > index_dip.x:  # Adjust condition based on the orientation of the hand
            raised_fingers += 1

        # Check other fingers
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]

        for tip in finger_tips:
            fingertip = hand_landmarks.landmark[tip]
            fingerpip = hand_landmarks.landmark[tip - 2]  # Use PIP joint for comparison
            if fingertip.y < fingerpip.y:  # Finger is raised if the TIP is above the PIP joint
                raised_fingers += 1

        return raised_fingers

    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        gesture_info = ""
        current_number_of_fingers = -1  # Default when no hand is detected

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if self.detect_start_gesture(hand_landmarks):
                    self.start_gesture_detected_frame_count = self.frames_to_hold_gesture

                if self.start_gesture_detected_frame_count > 0:
                    gesture_info = "Start Gesture Active"
                    if self.detect_new_number_gesture(hand_landmarks):
                        if self.consecutive_new_number_frames == 0:
                            self.new_number_gesture_detected = True  # Detect the start of a new gesture
                        self.consecutive_new_number_frames += 1  # Increment for consecutive frames
                        
                        # Once the gesture has been held for 15 frames, acknowledge it and reset
                        if self.consecutive_new_number_frames >= 25:
                            gesture_info += " - New Number Gesture Detected"
                            # Append a 0 to the numbers_sum list
                            self.numbers_sum.append(0)
                            self.current_sum_index += 1
                            self.consecutive_new_number_frames = 0  # Reset the counter for the next detection
                            self.new_number_gesture_detected = False  # Allow detection of a new gesture immediately
                    else:
                        self.consecutive_new_number_frames = 0  # Reset if not continuously detected
                        self.new_number_gesture_detected = False  # Reset to allow for a new gesture detection
                        
                    current_number_of_fingers = self.count_fingers(hand_landmarks)
                    if current_number_of_fingers == self.current_detected_number:
                        self.consecutive_frame_count += 1
                        if self.consecutive_frame_count == self.consecutive_frames_required:
                            # Add the current detected number to the sum at the current index
                            self.numbers_sum[self.current_sum_index] += current_number_of_fingers
                    else:
                        self.current_detected_number = current_number_of_fingers
                        self.consecutive_frame_count = 1
                else:
                    gesture_info = "Waiting for Start Gesture"
                    self.current_detected_number = None
                    self.consecutive_frame_count = 0

        if self.start_gesture_detected_frame_count > 0:
            self.start_gesture_detected_frame_count -= 1

        finger_count_display = f"Detected Fingers: {current_number_of_fingers}" if current_number_of_fingers >= 0 else ""
        return img, gesture_info, finger_count_display

def reset_numbers_sum():
    hand_gesture_recognition.numbers_sum = [0]  # Reset the numbers_sum list
    numbers_label.config(text="Sequence: [0]")  # Update the GUI to reflect the reset


def update_frame():
    global window, canvas, photo, hand_gesture_recognition, gesture_info_label, numbers_label, finger_count_label, q
    
    frame = q.tryGet()
    if frame is not None:
        img = frame.getCvFrame()
        img = cv2.flip(img, 1)
        img, gesture_info, finger_count_display = hand_gesture_recognition.process_frame(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        
        photo = ImageTk.PhotoImage(image=im_pil)
        
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        gesture_info_label.config(text=gesture_info)
        numbers_label.config(text="Sequence: " + str(hand_gesture_recognition.numbers_sum))
        finger_count_label.config(text=finger_count_display)
        
    window.after(10, update_frame)

if __name__ == "__main__":
    window = tk.Tk()
    window.title("Hand Gesture Recognition")
    
    canvas = tk.Canvas(window, width=640, height=480)
    canvas.pack()

    gesture_info_label = tk.Label(window, text="Gesture Info", font=('Helvetica', 12))
    gesture_info_label.pack()

    numbers_label = tk.Label(window, text="Numbers: ", font=('Helvetica', 12))
    numbers_label.pack()

    finger_count_label = tk.Label(window, text="Detected Fingers: ", font=('Helvetica', 12))
    finger_count_label.pack()

    btn_quit = tk.Button(window, text="Quit", command=window.destroy)
    btn_quit.pack(anchor=tk.CENTER, expand=True)

    btn_reset = tk.Button(window, text="Reset Sequence", command=reset_numbers_sum)
    btn_reset.pack(anchor=tk.CENTER, expand=True)

    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.createXLinkOut()
    xout.setStreamName("stream")
    cam.preview.link(xout.input)

    hand_gesture_recognition = HandGestureRecognition()

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="stream", maxSize=4, blocking=False)
        
        update_frame()
        
        window.mainloop()
