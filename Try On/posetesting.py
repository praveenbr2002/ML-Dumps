import cv2
import time
import mediapipe as mp
# we import the Twilio client from the dependency we just installed
from twilio.rest import Client
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For webcam input:
cap = cv2.VideoCapture(0)
fw = int(cap.get(3))
fh = int(cap.get(4))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 8.0, (fw, fh))
count = 0

def alertmsg():
    #the following line needs your Twilio Account SID and Auth Token
    client = Client("AC7b1804c6b337c4d84120f35f06e366ac", "6f1b9d5b76e4ff486b73dca0d478364a")
    client.messages.create(to="+919910581351", 
                    from_="+14708237934", 
                    body="Hello, Anomaly Detected!")
    # time.sleep(60)
    return client

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
    images = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    try:
        images.flags.writeable = False
        results = pose.process(images)
        landmark_value = results.pose_landmarks
        print(landmark_value.landmark)
        alertmsg()

        # Draw the pose annotation on the image.
        image = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        draw_mp = mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # print(draw_mp)
        # count += 1
        # if count == 1:
        #     #the following line needs your Twilio Account SID and Auth Token
        #     client = Client("AC7b1804c6b337c4d84120f35f06e366ac", "6f1b9d5b76e4ff486b73dca0d478364a")
        #     client.messages.create(to="+919910581351", 
        #                     from_="+14708237934", 
        #                     body="Hello, Anomaly Detected!")

        # frame is converted to hsv
        img = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
      
        # output the frame
        out.write(img)

        cv2.imshow('Pose Tracking Window', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
    except:
        pass
cap.release()