#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:07:28 2021

@author: bhajji
#python3 -m venv mp_env && source mp_env/bin/activate
"""

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

mp_drawinglip = mp.solutions.drawing_utils
PINK_COLOR = (203, 193, 255)
BLACK_COLOR = (17, 17, 17)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(color= BLACK_COLOR, thickness=2, circle_radius=1)
drawing_speclip = mp_drawinglip.DrawingSpec(color= PINK_COLOR, thickness=2, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    image_rows, image_cols, _ = image.shape
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec)

        mp_drawinglip.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONSLIP,
            landmark_drawing_spec = mp_drawinglip.DrawingSpec(colorlip= PINK_COLOR))

        
        # print(type(face_landmarks))
      #print(mp_face_mesh.FACE_CONNECTIONS, results.multi_face_landmarks)
#       [
#     print('x is', data_point.x, 'y is', data_point.y, 'z is', data_point.z,
#           'visibility is', data_point.visibility)
#     for data_point in face_landmarks.landmark
# ]

        # [print(face_landmarks.landmark[0])]
        x_coordinate = (face_landmarks.landmark[0].x * image_cols)
        y_coordinate = (face_landmarks.landmark[0].y * image_rows)
        print(x_coordinate, y_coordinate)
    #print(face_mesh)
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.findContours(img, cv2.CV_8UC1, cv2.Cha, offset=(0, 0))
    cv2.imshow('FaceMesh Tracking Window', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()