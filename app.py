import streamlit as st
import cv2
import numpy as np
from tensorflow import keras

# Set page title and description
st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")
st.title('Facial Emotion Recognition - BOCIL NI BOS')
st.write("Start the webcam and let's see what emotions you're feeling!")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model('facial-emotion-recognition-using-cnn.keras')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model with better error handling
model = load_model()
if model is None:
    st.stop()

# Add a button to start/stop the webcam
run = st.checkbox('Start Webcam')

# Create a placeholder for the webcam feed
video_placeholder = st.empty()

# Define emotions
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error('Failed to initialize webcam. Please check your camera connection.')
    st.stop()

try:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error('Failed to capture video frame.')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Process face and predict emotion
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            face_roi = np.expand_dims(face_roi / 255.0, axis=[0, -1])
            
            prediction = model.predict(face_roi)
            emotion_idx = np.argmax(prediction[0])
            emotion_label = f"{EMOTIONS[emotion_idx]} ({prediction[0][emotion_idx]*100:.1f}%)"

            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
finally:
    cap.release()
