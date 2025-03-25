# import the necessary packages
import datetime
import imutils
import time
import cv2

# reading from webcam
vs = cv2.VideoCapture(0)
# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:
	ret, frame = vs.read()
	frame = frame
	text = "No Object in Motion"
	# if the frame could not be grabbed, then we have reached the end video
	if frame is None:
		break
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (25, 25), 0)
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue
    
    # compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]
    
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 1000:
    			continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Object in motion"
	if text == "Object in motion":
    		color = (0, 255, 0)
	else:
    		color = (0, 0, 255)
    	# draw the text and timestamp on the frame
	cv2.putText(frame, "Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	# show the frame and record if the user presses a key
    
	cv2.imshow("Security Feed Window", frame)
	#cv2.imshow("Frame Delta Window", frameDelta)
	key = cv2.waitKey(1) & 0xFF
	time.sleep(0.020)
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()

