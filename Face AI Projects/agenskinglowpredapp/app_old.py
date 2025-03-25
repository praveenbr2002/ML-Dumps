# import the necessary packages
import cv2
import os
import random
import numpy as np
from PIL import Image
import streamlit as st
from mtcnn import MTCNN

############### Streamlit Front-End ################
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

logo = Image.open("Artifutech_logo.png")
st.sidebar.image(logo, width=240, output_format="auto")
st.sidebar.title("Artifutech Face AI App")
st.sidebar.warning("For better results please follow below rules for using App")
st.sidebar.success("1. Take pictures without any filters or in dark or in enough light.\n 2. Upload a picture of human clicked by camera at atleast 2 feet and atmost 4 feet distance.\n 3. Follow the instructions to predict age and glowness of skin.")

## Logo
image_logo = Image.open("logo2.jpeg")
st.image(image_logo, use_column_width=True)

# define the list of age buckets our age detector will predict
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-23)", "(25-35)",
	"(38-43)", "(48-53)", "(60-100)"]

# load our serialized face detector model from disk
#st.write("[INFO] loading face detector model...")
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our serialized age detector model from disk
#st.write("[INFO] loading age detector model...")
prototxtPath = "age_detector/age_deploy.prototxt"
weightsPath = "age_detector/age_net.caffemodel"
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

### Image Input ###
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Age & Face's Skin Glowness Prediction")
uploaded_file = st.file_uploader("Choose an image to check age of Human :", type=["png", "jpeg", "jpg"])
if st.checkbox("Show Image"):
    if uploaded_file is not None:
        try:
            images = Image.open(uploaded_file)
            img_array = np.array(images)## To read image in OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)## To read image in OpenCV
            st.image(images, caption='Uploaded Image.', use_column_width=True)
            
            st.warning("To predict the age and glowness of face, please select below checkbox :")
            
            #### Function for face part cropping ######
            source_dir = img_array#cv2.imread(img)#cv2.imread(uploaded_file)
            ####### For Face Cropping ######################
            detector = MTCNN()
            mode=1   
            face_source_dir = source_dir#"./face/"
            dest_dir = "./cropped_face/"
            
            #Function to crop the face box
            def crop_face_image(face_source_dir, dest_dir, mode):
                if os.path.isdir(dest_dir)==False:
                    os.mkdir(dest_dir)
                detector = MTCNN()
                img = source_dir#cv2.imread(source_dir)
                data=detector.detect_faces(img)
                if mode==1:  #detect the box with the largest area
                    for i, faces in enumerate(data): # iterate through all the faces found
                        box=faces['box']  # get the box for each face                
                        biggest=0                    
                        area = box[3]  * box[2]
                        if area>biggest:
                            biggest=area
                            bbox=box 
                    bbox[0]= 0 if bbox[0]<0 else bbox[0]
                    bbox[1]= 0 if bbox[1]<0 else bbox[1]
                    img=img[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]] 
                    cv2.imwrite("./cropped_face/cropped_face.png", img)
            
                return 1
            
            ########### Directories for forehead ################
            forehead_source_dir = "./cropped_face/cropped_face.png"#source_dir
            forehead_dest_dir = "./cropped_forehead/"
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            
            #Function to crop the Forehead Part
            def crop_forehead_image(forehead_source_dir, forehead_dest_dir, mode):
                if os.path.isdir(forehead_dest_dir)==False:
                    os.mkdir(forehead_dest_dir)
            
                imgs = cv2.imread("./cropped_face/cropped_face.png")#source_dir#
                gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray)
                for (eye_startX, eye_startY, eye_width, eye_height) in eyes:
                    cv2.rectangle(imgs, (eye_startX, eye_startY), (eye_startX + eye_width, eye_startY + eye_height), (0, 255, 0), 2)
            
                if mode==1:  #detect the box with the largest area
                    for i, eye in enumerate(eyes): # iterate through all the eye found
                        biggest=0                    
                        area = eye_height * eye_width
                        if area>biggest:
                            biggest=area
                            
                    eye_startX= 0 if eye_startX<0 else eye_startX
                    eye_startY= 0 if eye_startY<0 else eye_startY
                    #img=img[eye_startY: eye_startY+eye_height+10, eye_startX: eye_startX+ eye_width]
                    imgs=imgs[eye_startY-40: eye_startY-14, eye_startX+10: eye_startX+60]
                    cv2.imwrite("./cropped_forehead/cropped_forehead.png", imgs)
                        
                return 1
            
            ######### Directories for cheeks ############
            cheek_source_dir = "./cropped_face/cropped_face.png"#source_dir
            cheek_dest_dir = "./cropped_cheek/"
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            
            #Function to crop the cheek Part
            def crop_cheek_image(cheek_source_dir, cheek_dest_dir, mode):
                if os.path.isdir(cheek_dest_dir)==False:
                    os.mkdir(cheek_dest_dir)
                
                imgsc = cv2.imread("./cropped_face/cropped_face.png")#source_dir
                gray = cv2.cvtColor(imgsc, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray)
                for (eye_startX, eye_startY, eye_width, eye_height) in eyes:
                    cv2.rectangle(imgsc, (eye_startX, eye_startY), (eye_startX + eye_width, eye_startY + eye_height), (0, 255, 0), 2)
            
                if mode==1:  #detect the box with the largest area
                    for i, eye in enumerate(eyes): # iterate through all the eye found
                        biggest=0                    
                        area = eye_height * eye_width
                        if area>biggest:
                            biggest=area
                            
                    eye_startX= 0 if eye_startX<0 else eye_startX
                    eye_startY= 0 if eye_startY<0 else eye_startY
                    #img=img[eye_startY: eye_startY+eye_height+10, eye_startX: eye_startX+ eye_width]
                    imgsc=imgsc[eye_startY+eye_height+10: eye_startY+eye_height+36, eye_startX: eye_startX + 50]
                    cv2.imwrite("./cropped_cheek/cropped_cheek.png", imgsc)
                        
                return 1
            ############# Age Prediction ####
            if st.checkbox("Age Prediction"):
                try:
                    ## Function Call for face
                    crop_face_image(face_source_dir, dest_dir, mode)
                    
                    # Function Call for forehead 
                    crop_forehead_image(forehead_source_dir, forehead_dest_dir, mode)
                    
                    # Function Call for cheeks
                    crop_cheek_image(cheek_source_dir, cheek_dest_dir, mode)
                    # load the input image and construct an input blob for the image
                    image = img_array#cv2.imread(args["image"])
                    (h, w) = image.shape[:2]
                    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                    	(104.0, 177.0, 123.0))
                    # pass the blob through the network and obtain the face detections
                    st.write("Predicting age of human..!")
                    faceNet.setInput(blob)
                    detections = faceNet.forward()
                    
                    # loop over the detections
                    for i in range(0, detections.shape[2]):
                    	# extract the confidence (i.e., probability) associated with the
                    	# prediction
                    	confidence = detections[0, 0, i, 2]
                    	# filter out weak detections by ensuring the confidence is
                    	# greater than the minimum confidence
                    	if confidence > 0.6:
                    		# compute the (x, y)-coordinates of the bounding box for the
                    		# object
                    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    		(startX, startY, endX, endY) = box.astype("int")
                    		# extract the ROI of the face and then construct a blob from
                    		# *only* the face ROI
                    		face = image[startY:endY, startX:endX]
                    		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                    			(78.4263377603, 87.7689143744, 114.895847746),
                    			swapRB=False)
                            # make predictions on the age and find the age bucket with
                    		# the largest corresponding probability
                    		ageNet.setInput(faceBlob)
                    		preds = ageNet.forward()
                    		i = preds[0].argmax()
                    		age = AGE_BUCKETS[i]
                    		ageConfidence = preds[0][i]
                    		# display the predicted age to our terminal
                    		text = "{} Years".format(age, ageConfidence * 100)
                    		st.write("Age Range : ", text)
                    		# draw the bounding box of the face along with the associated
                    		# predicted age
                    		#y = startY - 10 if startY - 10 > 10 else startY + 10
                    		#cv2.rectangle(image, (startX, startY), (endX, endY),
                    		#	(0, 255, 0), 2)
                    		#cv2.putText(image, text, (startX, y),
                    		#	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        
                    #new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #st.image(new_img, use_column_width=True)
                except:
                    st.warning("Please, upload another file!")
                    
            if st.checkbox("Skin Glowness Score"):
                try:
                    st.write("Predicting face's skin glowness score..!")
                    ## Function Call for face
                    crop_face_image(face_source_dir, dest_dir, mode)
                    
                    # Function Call for forehead 
                    crop_forehead_image(forehead_source_dir, forehead_dest_dir, mode)
                    
                    # Function Call for cheeks
                    crop_cheek_image(cheek_source_dir, cheek_dest_dir, mode)
                    
                    ##Forehead Part
                    img_forehead = cv2.imread('./cropped_forehead/cropped_forehead.png')
                    img_forehead = cv2.cvtColor(img_forehead, cv2.COLOR_BGR2RGB)
                    img_forehead = cv2.GaussianBlur(img_forehead, (5,5), 0)
                    
                    img_forehead_g = cv2.cvtColor(img_forehead, cv2.COLOR_RGB2GRAY)
                    sobely_img_forehead = cv2.Sobel(img_forehead_g, cv2.CV_8UC1, 0, 1, ksize=5)
                    fix_hy, fix_wy = sobely_img_forehead.shape
                    param_img_forehead_1 = cv2.sumElems(sobely_img_forehead)[0]/(fix_hy * fix_wy)
                    #st.write("Value of Forehead Part is ", param_img_forehead_1)
                    
                    ##cheeks Part
                    img_cheeks = cv2.imread('./cropped_cheek/cropped_cheek.png')
                    img_cheeks = cv2.cvtColor(img_cheeks, cv2.COLOR_BGR2RGB)
                    img_cheeks = cv2.GaussianBlur(img_cheeks, (5,5), 0)
                    
                    img_cheeks_g = cv2.cvtColor(img_cheeks, cv2.COLOR_RGB2GRAY)
                    sobely_img_cheeks = cv2.Sobel(img_cheeks_g, cv2.CV_8UC1, 0, 1, ksize=5)
                    chk_hy, chk_wy = sobely_img_cheeks.shape
                    param_img_cheeks_1 = cv2.sumElems(sobely_img_cheeks)[0]/(chk_hy * chk_wy)
                    #st.write("Value of Cheeks Part is ", param_img_cheeks_1)
                    
                    avg_skin_value = (param_img_forehead_1 + param_img_cheeks_1)/2
                    #st.write("Average skin value is ", avg_skin_value)
                    
                    ############ Rules for Skin Glowness ################
                    if param_img_forehead_1 > param_img_cheeks_1:
                        if (param_img_forehead_1 - avg_skin_value) > 0 and (param_img_forehead_1 - avg_skin_value) <=5:
                            st.write("Face Skin Glowness Score is ", random.randint(86,94))
                        else:
                            if (param_img_forehead_1 - avg_skin_value) > 5 and (param_img_forehead_1 - avg_skin_value) <=10:
                                st.write("Face Skin Glowness Score is ", random.randint(78,85))
                            else:
                                if (param_img_forehead_1 - avg_skin_value) > 10 and (param_img_forehead_1 - avg_skin_value) <=15:
                                    st.write("Face Skin Glowness Score is ", random.randint(71,77))
                                else:
                                    st.write("Face Skin Glowness Score is ", random.randint(61,70))
                    else:
                        if (param_img_cheeks_1 - avg_skin_value) > 0 and (param_img_cheeks_1 - avg_skin_value) <=5:
                            st.write("Face Skin Glowness Score is ", random.randint(86,94))
                        else:
                            if (param_img_cheeks_1 - avg_skin_value) > 5 and (param_img_cheeks_1 - avg_skin_value) <=10:
                                st.write("Face Skin Glowness Score is ", random.randint(78,85))
                            else:
                                if (param_img_cheeks_1 - avg_skin_value) > 10 and (param_img_cheeks_1 - avg_skin_value) <=15:
                                    st.write("Face Skin Glowness Score is ", random.randint(71,77))
                                else:
                                    st.write("Face Skin Glowness Score is ", random.randint(61,70))
                
                except:
                    st.warning("Please, upload another file!")

        except:
            st.warning("Please, upload another file!")
