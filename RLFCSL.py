import cv2
import mediapipe as mp
import streamlit as st

# Used to convert protobuf message to a dictionary.
from google.protobuf.json_format import MessageToDict

def detect_hands_finger_Count(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # For webcam input:

    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2
        )
    
    # Draw the hand annotations on the image.

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)


    # Finger Count
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
            #   considered raised.
            # Thumb: TIP x position must be greater or lower than IP x position,
            #   deppeding on hand label.
            if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                fingerCount = fingerCount + 1
            elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                fingerCount = fingerCount + 1

            # Other fingers: TIP y position must be lower than PIP y position,
            #   as image origin is in the upper left corner.
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
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Display finger count
    cv2.putText(img, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)


    #Hand Detection

    if results.multi_hand_landmarks:
        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
            # Display 'Both Hands' on the image
            cv2.putText(img, 'Both Hands', (250, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

        # If any hand present
        else:
            for i in results.multi_handedness:

                # Return whether it is Right or Left Hand
                label = MessageToDict(i)['classification'][0]['label']
                print(label)
                if label == 'Left':
                    # Display 'Left Hand' on
                    # left side of window
                    cv2.putText(img, label + ' Hand', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

                if label == 'Right':
                    # Display 'Left Hand'
                    # on left side of window
                    cv2.putText(img, label + ' Hand', (460, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    st.title("Hand Detection and Finger Count with Streamlit")
    st.write("Press 'Start' to begin hand detection.")
    st.write("Press 'Esc' to stop hand detection.")

    # Create a placeholder to display the webcam feed
    placeholder = st.empty()

    # Start capturing video from webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Process the frame and get the processed image
        processed_img = detect_hands_finger_Count(img)

        # Display the processed image sing the placeholder
        placeholder.image(processed_img, channels="RGB")

        # Display the image
        cv2.imshow('MediaPipe Hands', img)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()