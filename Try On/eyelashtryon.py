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
#mp_face_point = mp.solutions.face_mesh_test

BLACK_COLOR = (17, 17, 17)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(color= BLACK_COLOR, thickness=2, circle_radius=1)
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
        for ids, lm in enumerate(face_landmarks.landmark):
            h, w, c = image.shape
            #print(ids, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            print(ids, cx, cy)
            #cv2.circle(image, (cx,cy), 5, (255,0,0), cv2.FILLED)
        #x_coordinate = (face_landmarks.landmark[0].x * image_cols)
        #y_coordinate = (face_landmarks.landmark[0].y * image_rows)
        #print(x_coordinate, y_coordinate)
    
    cv2.imshow('Eyeliner Tryon Window', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()