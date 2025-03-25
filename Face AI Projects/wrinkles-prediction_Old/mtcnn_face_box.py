import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            startX = bounding_box[0]
            startY = bounding_box[1]
            endX = bounding_box[0] + bounding_box[2]
            endY = bounding_box[1] + bounding_box[3]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
    #display resulting frame
    cv2.imshow('frame',frame)
    cv2.imwrite('./dataset_raw/cropped_raw_img.png',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()