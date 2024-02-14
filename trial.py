import numpy as np
import streamlit as st
import pandas as pd
from keras.models import load_model
from PIL import Image

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
    image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image is not None:
        image = Image.open(image)
        image = np.array(image)
        gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]) # Convert ke grayscale
        faces = None # Proses deteksi wajah bisa diganti dengan algoritma lain jika diperlukan
        try:
            for (p, q, r, s) in faces:
                face_image = gray[q:q+s, p:p+r]
                resized_image = np.array(Image.fromarray(face_image).resize((48, 48)))  # Resize gambar
                img = resized_image.reshape(1, 48, 48, 1) / 255.0
                pred = model.predict(img)
                emotion_label = labels[pred.argmax()]
                
                recommended_songs = recommend_songs(emotion_label)
                
                st.image(resized_image, channels="GRAY")

                st.subheader("Recommended Songs For You:")
                st.markdown("<ul>", unsafe_allow_html=True)
                for index, row in recommended_songs.iterrows():
                    text = f"<li>{row['Song']} - {row['Artist']}</li>"
                    st.markdown(text, unsafe_allow_html=True)
                st.markdown("</ul>", unsafe_allow_html=True)
                
        except Exception as e:
            print(e)
