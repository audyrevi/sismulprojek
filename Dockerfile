FROM python:3.11

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y