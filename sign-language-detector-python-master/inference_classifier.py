import asyncio
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Dictionary mapping gesture labels to their meanings
labels_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: 'sorry', 5: 'Thank you'}

# Function to speak out the predicted text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to process video frames and perform gesture recognition
async def process_video():
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)  # Process the frame with MediaPipe Hands

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # Image to draw
                    hand_landmarks,  # Model output
                    mp_hands.HAND_CONNECTIONS,  # Hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []  # Initialize data_aux list
            x_ = []  # Initialize x_ list
            y_ = []  # Initialize y_ list

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Predict gesture label and calculate accuracy
            prediction = model.predict([np.asarray(data_aux)])
            accuracy = model.predict_proba([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            accuracy_text = f"Accuracy: {accuracy[0][int(prediction[0])] * 100:.2f}%"

            # Speak out the predicted text
            speak(predicted_character)

            # Draw rectangle and text on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"{predicted_character} ({accuracy_text})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# Start the event loop and run the process_video coroutine
async def main():
    await process_video()

asyncio.run(main())

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
q