# import the necessary packages
import numpy as np
import cv2
import dlib
import imutils

webcam = True

cap = cv2.VideoCapture(0)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def empty(a):
    pass
cv2.namedWindow('Lipstick Try On')
cv2.resizeWindow('Lipstick Try On', 640, 240)
cv2.createTrackbar("BLUE", 'Lipstick Try On', 0, 255, empty)
cv2.createTrackbar("GREEN", 'Lipstick Try On', 0, 255, empty)
cv2.createTrackbar("RED", 'Lipstick Try On', 0, 255, empty)

def createBox(image, points, masked=False, Cropped=True):
    if masked:
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, [points], (255,255,255))
        image = cv2.bitwise_and(image, mask)
        #cv2.imshow("mask", image)
    
    if Cropped:    
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        imgCrop = image[y:y+h,x:x+w]
        imgCrop = imutils.resize(imgCrop, width=500)
        return imgCrop
    else:
        return mask

while True:
    
    if webcam: ret, image = cap.read()
    else: image = cv2.imread('images/100.png')
    image = imutils.resize(image, width=500)
    imgoriginal = image.copy()
    
    imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    faces = detector(imggray)
    
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        #imgoriginal = cv2.rectangle(image, (x1,y1), (x2, y2), (0,255,0), 2)
        landmarks = predictor(imggray, face)
        mypoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mypoints.append([x,y])
            #cv2.circle(imgoriginal, (x,y), 2, (255,0,0), cv2.FILLED)
            #cv2.putText(imgoriginal, str(n), (x,y+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1)
        
        mypoints = np.array(mypoints)
        imgLips = createBox(image, mypoints[48:61], masked=True, Cropped=False)
        #cv2.imshow("Lips", imgLips)
        
        imgColorLips = np.zeros_like(imgLips)
        BLUE = cv2.getTrackbarPos("BLUE", 'Lipstick Try On')
        GREEN = cv2.getTrackbarPos("GREEN", 'Lipstick Try On')
        RED = cv2.getTrackbarPos("RED", 'Lipstick Try On')
        imgColorLips[:] = [BLUE,GREEN,RED]
        imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips, (7,7),10)
        imgoriginalGray = cv2.cvtColor(imgoriginal, cv2.COLOR_BGR2GRAY)
        imgoriginalGray = cv2.cvtColor(imgoriginalGray, cv2.COLOR_GRAY2BGR)
        imgColorLips = cv2.addWeighted(imgoriginalGray, 1, imgColorLips, 0.4, 0)
        cv2.imshow("Lipstick Try On", imgColorLips)
        
    #cv2.imshow("OriginalImage", imgoriginal)
    cv2.waitKey(1)
