import cv2
import numpy as np
import streamlit as st
import pandas as pd
from keras.models import load_model

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

clicked = st.button("Click to Detect Emotion and Recommend Songs")

if clicked:
    webcam = cv2.VideoCapture(0)
    _, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            image = cv2.resize(image, (48, 48))
            img = np.array(image).reshape(1, 48, 48, 1) / 255.0
            pred = model.predict(img)
            emotion_label = labels[pred.argmax()]
            
            recommended_songs = recommend_songs(emotion_label)
            
            # Menambahkan teks emosi di atas video
            cv2.putText(im, f"Emotion: {emotion_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            st.image(im, channels="BGR")

            st.subheader("Recommended Songs For You:")
            st.markdown("<ul>", unsafe_allow_html=True)
            for index, row in recommended_songs.iterrows():
                text = f"<li>{row['Song']} - {row['Artist']}</li>"
                st.markdown(text, unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
                
    except cv2.error:
        pass
