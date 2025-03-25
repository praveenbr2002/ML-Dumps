import cv2
import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
logo = Image.open("Artifutech_logo.png")
st.sidebar.image(logo, width=299, output_format="auto")
st.sidebar.title("Artifutech Face AI App")

st.title("Webcam Face Detection")
FRAME_WINDOW = st.image([])

def capture_face(video_capture):
    while(True):
        ret, frame = video_capture.read()
        
        FRAME_WINDOW.image(frame[:, :, ::-1])
        
        # face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if len(face_locations) > 0:
            video_capture.release()
            return frame

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))

def recognize_frame(frame):
    # convert COLOR_BGR2RGB
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 153), 2)
        return frame[:, :, ::-1]

if __name__ == "__main__":
    while(True):
        try:
            video_capture = cv2.VideoCapture(0)
            frame = capture_face(video_capture)
            frame = recognize_frame(frame)
            FRAME_WINDOW.image(frame)
            
        except:pass