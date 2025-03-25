import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

#Eye HaarCascade Model
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Realtime camera
cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    for (eye_startX, eye_startY, eye_width, eye_height) in eyes:
        cv2.rectangle(img, (eye_startX, eye_startY), (eye_startX + eye_width, eye_startY + eye_height + 8), (0, 255, 0), 2)

    #display resulting eye frame
    cv2.imshow('img', img)
    cv2.imwrite('./eye/eye_crop.png', img)
    #Use MTCNN to detect faces frame by frame
    __, frame = cap.read()
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            startX = bounding_box[0]
            startY = bounding_box[1]
            endX = bounding_box[0] + bounding_box[2]
            endY = bounding_box[1] + bounding_box[3]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
    #display resulting face frame
    cv2.imshow('frame',frame)
    cv2.imwrite('./face/cropped_raw_img.png',frame)
        
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()