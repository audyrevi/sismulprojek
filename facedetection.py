import cv2
import numpy as np
import streamlit as st
import pandas as pd
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Membaca dataset lagu
data = pd.read_csv('datasetlagu.csv')

def filter_songs_by_emotion(emotion):
    filtered_songs = data[data['Emotion'] == emotion]
    return filtered_songs[['Artist', 'Song', 'Emotion']]

# Memuat model deteksi emosi
model = load_model('best_model.h5')

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def recommend_songs(emotion_label):
    recommended_songs = filter_songs_by_emotion(emotion_label)
    return recommended_songs

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video transformer untuk deteksi emosi wajah
class FaceEmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.is_initialized = False

    def initialize(self):
        self.is_initialized = True

    def transform(self, frame):
        if not self.is_initialized:
            self.initialize()

        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        try:
            for (p, q, r, s) in faces:
                face_image = gray[q:q+s, p:p+r]
                face_image = cv2.resize(face_image, (48, 48))
                img = np.array(face_image).reshape(1, 48, 48, 1) / 255.0
                pred = model.predict(img)
                emotion_label = labels[pred.argmax()]

                recommended_songs = recommend_songs(emotion_label)

                # Menambahkan teks emosi di atas video
                cv2.putText(image, f"Emotion: {emotion_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                for index, row in recommended_songs.iterrows():
                    cv2.putText(image, f"{row['Song']} - {row['Artist']}", (20, 80 + (index * 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except cv2.error:
            pass

        return image

#UI
st.markdown("<h1 style=' color: #800000; text-align: center;'>Welcome to our Music Recommendation Based on Face Emotion Recognition App! </h1>", unsafe_allow_html=True)
st.markdown("<p style=' color: #808080; font-size: 12px; text-align: center; '>Upload a photo of your face, and we'll recommend songs based on your detected emotion</p>", unsafe_allow_html=True)

#Spasi
st.markdown("<br>", unsafe_allow_html=True)

#Tampilan button
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #808080;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            text-align: center;
            display: block;
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 style=' text-align: center;'>Live Emotion Detection and Song Recommendation</h2>", unsafe_allow_html=True)

webrtc_streamer(key="emotion", video_transformer_factory=FaceEmotionTransformer)
