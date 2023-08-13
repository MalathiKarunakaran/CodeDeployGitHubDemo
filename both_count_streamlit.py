import cv2
import mediapipe as mp
import streamlit as st
from google.protobuf.json_format import MessageToDict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def main():
    st.title('Hand Tracking and Finger Counting')
    st.write("Raise your hand in front of the camera to see the finger count.")

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            if not success:
                st.warning("Error: Webcam feed not available.")
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Initially set finger count to 0 for each cap
            fingerCount = 0

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:
                    # Get hand index to check label (left or right)
                    handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                    handLabel = results.multi_handedness[handIndex].classification[0].label

                    # Set variable to keep landmarks positions (x and y)
                    handLandmarks = []

                    # Fill list with x and y positions of each landmark
                    for landmarks in hand_landmarks.landmark:
                        handLandmarks.append([landmarks.x, landmarks.y])

                    # Test conditions for each finger: Count is increased if finger is
                    # considered raised.
                    # Thumb: TIP x position must be greater or lower than IP x position,
                    # depending on hand label.
                    if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                        fingerCount = fingerCount + 1
                    elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                        fingerCount = fingerCount + 1

                    # Other fingers: TIP y position must be lower than PIP y position,
                    # as image origin is in the upper left corner.
                    if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                        fingerCount = fingerCount + 1
                    if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                        fingerCount = fingerCount + 1
                    if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                        fingerCount = fingerCount + 1
                    if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                        fingerCount = fingerCount + 1

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Display finger count and hand label in the Streamlit app
            st.image(image, channels='RGB', use_column_width=True)
            st.write(f"Finger Count: {fingerCount}")

            # Exit the loop if the 'Esc' key is pressed
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

if __name__ == "__main__":
    main()
