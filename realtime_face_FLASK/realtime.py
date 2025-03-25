# Core Pkgs
import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os


face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')

def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
				 cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img,faces 


def detect_eyes(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
	for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

def detect_smiles(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect Smiles
	smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the Smiles
	for (x, y, w, h) in smiles:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img

def main():
	"""Face Detection App"""

	st.title("Face Detection App")
	st.text("Build with Streamlit and OpenCV")

	activities = ["Detection"]
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'Detection':
		st.subheader("Face Detection")
        
        image_file = cv2.VideoCapture(0)
        while(True):
    		if image_file is not None:
    			ret, our_image = image_file.read()
    			st.text("Original Image")
    			# st.write(type(our_image))
    			st.image(our_image)
                
    		# Face Detection
    		task = ["Faces","Smiles","Eyes"]
    		feature_choice = st.sidebar.selectbox("Find Features",task)
    		if st.button("Process"):
    
    			if feature_choice == 'Faces':
    				result_img,result_faces = detect_faces(our_image)
    				st.image(result_img)
    
    				st.success("Found {} faces".format(len(result_faces)))
    			elif feature_choice == 'Smiles':
    				result_img = detect_smiles(our_image)
    				st.image(result_img)
    
    
    			elif feature_choice == 'Eyes':
    				result_img = detect_eyes(our_image)
    				st.image(result_img)
    

if __name__ == '__main__':
		main()