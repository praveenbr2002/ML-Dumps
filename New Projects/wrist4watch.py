#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:48:11 2021

@author: bhajji
"""

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

######### Function
def findPosition():
    lmList = []
    if results.pose_landmarks:
        myHand = results.pose_landmarks.landmark
        for id, lm in enumerate(myHand):
            # print(id, lm)
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id, cx, cy)
            lmList.append([id, cx, cy])
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
 
    return lmList

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
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
    results = pose.process(image)
    lmList = findPosition()
    
    if len(lmList) != 0:
        print("Right Wrist", lmList[15], "Left Wrist", lmList[16])

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #mp_drawing.draw_landmarks(
     #   image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Wrist LandMark Point', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()