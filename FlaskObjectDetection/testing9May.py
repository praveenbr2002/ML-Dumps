import cv2
import numpy as np
# import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # here used for detection of my red bottle cap, adjust as per your requirement
    # hsv hue and sat value
    lower_red = np.array([100,0,0])
    upper_red = np.array([255,255,255])
    
    mask = cv2.inRange(hsv,lower_red,upper_red)
    res = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()