# import the necessary packages
import os
import cv2
import random
import numpy as np
from PIL import Image
import streamlit as st
from mtcnn.mtcnn import MTCNN

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

## MTCNN Face
detector = MTCNN()

### Image Input ###
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Dark Spots Prediction")
uploaded_file = st.file_uploader("Choose an image to predict the dark spots score :", type=["png", "jpeg", "jpg"])
if st.checkbox("Show Image"):
    if uploaded_file is not None:
        try:
            images = Image.open(uploaded_file)
            img_array = np.array(images)## To read image in OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)## To read image in OpenCV
            st.image(images, caption='Uploaded Image.', use_column_width=True)
            
            st.warning("To predict the dark spots score on face, please select below checkbox :")
            
            #### Function for face part cropping ######
            source_dir = img_array
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
                    imgs=imgs[eye_startY-40: eye_startY-14, eye_startX+10: eye_startX+50]
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
                    imgsc=imgsc[eye_startY+eye_height+10: eye_startY+eye_height+36, eye_startX: eye_startX + 40]
                    cv2.imwrite("./cropped_cheek/cropped_cheek.png", imgsc)
                        
                return 1
            ############# Dark spot Prediction ####
            if st.checkbox("Dark spots Prediction Score"):
                try:
                    ## Function Call for face
                    crop_face_image(face_source_dir, dest_dir, mode)
                    
                    # Function Call for forehead 
                    crop_forehead_image(forehead_source_dir, forehead_dest_dir, mode)
                    
                    # Function Call for cheeks
                    crop_cheek_image(cheek_source_dir, cheek_dest_dir, mode)
                    # load the input image and construct an input blob for the image
                    images = img_array      #cv2.imread(args["image"])
                    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                    
                    ##Forehead Part
                    img_Forehead = cv2.imread('./cropped_forehead/cropped_forehead.png')
                    img_Forehead = cv2.cvtColor(img_Forehead, cv2.COLOR_RGB2BGR)
                    img_Forehead_g = cv2.cvtColor(img_Forehead, cv2.COLOR_BGR2GRAY)
                    #st.write(img_Forehead_g)
                    #st.image(img_Forehead, caption='Forehead.', use_column_width=True)
                    
                    ##Cheeks Part
                    img_Cheeks = cv2.imread('./cropped_cheek/cropped_cheek.png')
                    img_Cheeks = cv2.cvtColor(img_Cheeks, cv2.COLOR_RGB2BGR)
                    img_Cheeks_g = cv2.cvtColor(img_Cheeks, cv2.COLOR_BGR2GRAY)
                    #st.write(img_Cheeks_g)
                    #st.image(img_Cheeks, caption='Cheeks.', use_column_width=True)
                    
                    ### Calculating average skin value #####
                    forehead_max = int(np.max(img_Forehead_g))
                    forehead_min = int(np.min(img_Forehead_g))
                    cheek_max = int(np.max(img_Cheeks_g))
                    cheek_min = int(np.min(img_Cheeks_g))
                    avg_max_value = (forehead_max + cheek_max)/2
                    avg_min_value = (forehead_min + cheek_min)/2
                    #st.write(avg_max_value, avg_min_value)
                    ############ Rules for Skin Glowness ################
                    if avg_min_value >= 140:
                        st.subheader("Predicting Dark Spots Score..!")
                        st.write("Where 100 means dark spot is available in large amount and 0 means no dark spot present at face.")
                        if avg_max_value >= 195:
                            st.write("Predicted Dark Spots Score is ", random.randint(30,40))
                        else:
                            if avg_max_value < 195 and avg_max_value >= 180:
                                st.write("Predicted Dark Spots Score is ", random.randint(45,50))
                            else:
                                if avg_max_value < 180 and avg_max_value >= 170:
                                    st.write("Predicted Dark Spots Score is ", random.randint(50,55))
                                else:
                                    if avg_max_value < 170 and avg_max_value >= 160:
                                        st.write("Predicted Dark Spots Score is ", random.randint(55,60))
                                    else:
                                        if avg_max_value < 160 and avg_max_value >= 150:
                                            st.write("Predicted Dark Spots Score is ", random.randint(61,66))
                                        else:
                                            st.write("Predicted Dark Spots Score is ", random.randint(67,80))
                    else:
                        st.subheader("Predicting Dark Spots Score..!")
                        st.write("Where 100 means dark spot is available in large amount and 0 means no dark spot present at face.")
                        if avg_min_value < 140 and avg_min_value >= 125:
                            st.write("Predicted Dark Spots Score is ", random.randint(45,50))
                        else:
                            if avg_min_value < 125 and avg_min_value >= 115:
                                st.write("Predicted Dark Spots Score is ", random.randint(50,55))
                            else:
                                if avg_min_value < 115 and avg_min_value >= 105:
                                    st.write("Predicted Dark Spots Score is ", random.randint(55,60))
                                else:
                                    if avg_min_value < 105 and avg_min_value >= 90:
                                        st.write("Predicted Dark Spots Score is ", random.randint(60,65))
                                    else:
                                        if avg_min_value < 90 and avg_min_value >= 80:
                                            st.write("Predicted Dark Spots Score is ", random.randint(65,70))
                                        else:
                                            st.write("Predicted Dark Spots Score is ", random.randint(70,80))

                except:
                    st.warning("Please, upload another file!")

        except:
            st.warning("Please, upload another file!")
