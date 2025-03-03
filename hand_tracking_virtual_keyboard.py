# vr_hand_tracker.py
import cv2
import mediapipe as mp
import numpy as np
from virtual_keyboard import draw_keyboard, detect_key_press, keyboard_layout
from utils import detect_stickers, initialize_kalman

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_vr_camera_feed():
    """Placeholder for VR headset camera feed (e.g., Quest 2 passthrough)."""
    cap = cv2.VideoCapture(0)  # Test with webcam; replace with VR SDK
    if not cap.isOpened():
        print("Failed to open camera. Ensure VR headset or webcam is connected.")
        exit(1)
    return cap

def render_to_vr(image):
    """Placeholder for VR rendering (e.g., Oculus TextureSwapChain)."""
    cv2.imshow('VR Hand Tracking', image)  # Replace with VR SDK rendering

def main():
    """Main loop for VR hand tracking with sticker-enhanced MediaPipe."""
    cap = get_vr_camera_feed()
    kfs = [initialize_kalman() for _ in range(5)]  # Kalman filters for fingertips

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read VR camera frame.")
                break

            # Process frame with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw virtual keyboard
            image = draw_keyboard(image)

            # Detect stickers for fingertip enhancement
            sticker_positions = detect_stickers(image)  # Returns list of (x, y) coords

            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Enhance fingertip positions with stickers
                    if sticker_positions:
                        for i, (sx, sy) in enumerate(sticker_positions[:5]):  # Limit to 5 fingertips
                            tip_idx = mp_hands.HandLandmark.INDEX_FINGER_TIP + i * 4
                            # Convert sticker pixel coords to normalized MediaPipe coords
                            sticker_x = sx / image.shape[1]
                            sticker_y = sy / image.shape[0]
                            mp_tip = hand_landmarks.landmark[tip_idx]
                            # Weighted average (favor sticker position)
                            z = np.array([
                                sticker_x * 0.8 + mp_tip.x * 0.2,
                                sticker_y * 0.8 + mp_tip.y * 0.2,
                                mp_tip.z  # No depth from stickers yet
                            ])
                            kfs[i].predict()
                            kfs[i].update(z)
                            hand_landmarks.landmark[tip_idx].x, \
                            hand_landmarks.landmark[tip_idx].y, \
                            hand_landmarks.landmark[tip_idx].z = kfs[i].x

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Detect key presses
                    pressed_keys = detect_key_press(hand_landmarks, image, keyboard_layout)
                    if pressed_keys:
                        print(f"Pressed: {pressed_keys}")

            # Render to VR
            render_to_vr(image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
