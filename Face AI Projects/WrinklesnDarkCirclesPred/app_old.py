import cv2
import os
import numpy as np
import streamlit as st
from mtcnn import MTCNN
from PIL import Image
###########################################################################
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

logo = Image.open("Artifutech_logo.png")
st.sidebar.image(logo, width=200, output_format="auto")
st.sidebar.title("Artifutech Face AI App")
st.sidebar.warning("For better results please follow below rules for using App")
st.sidebar.success("1. Take pictures without any filters, glasses, beard or in dark or in enough light.\n 2. Upload a picture of human clicked by camera at atleast 2 feet and atmost 4 feet distance.\n 3. Follow the instructions to show image and predict wrinkles & dark circles.")

## Logo
image_logo = Image.open("logo2.jpeg")
st.image(image_logo, use_column_width=True)

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an image to check Wrinkles & Dark Circles on Human Face:", type=["png", "jpeg", "jpg"])
if st.checkbox("Show Image"):
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            #else:
            #   st.error("Please, Upload a valid file!")

            ############################################################################################################################
            st.warning("To check percentage of Wrinkles and Dark circles Prediction. Please click in below box.")
            ########### Functions to crop face & face part #############
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
            
            ######### Directories & Eye HaarCascade Model #########
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            eye_source_dir = "./cropped_face/cropped_face.png"#source_dir
            eye_dest_dir = "./cropped_eye/"
            
            ########## Function to crop the EYE box #########
            def crop_eye_image(eye_source_dir, eye_dest_dir, mode):
                if os.path.isdir(eye_dest_dir)==False:
                    os.mkdir(eye_dest_dir)
            
                img = cv2.imread("./cropped_face/cropped_face.png")#source_dir#
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray)
                for (eye_startX, eye_startY, eye_width, eye_height) in eyes:
                    cv2.rectangle(img, (eye_startX, eye_startY), (eye_startX + eye_width, eye_startY + eye_height + 10), (0, 255, 0), 2)
            
                if mode==1:  #detect the box with the largest area
                    for i, eye in enumerate(eyes): # iterate through all the eye found
                        biggest=0                    
                        area = eye_height  * eye_width
                        if area>biggest:
                            biggest=area
                            
                    eye_startX= 0 if eye_startX<0 else eye_startX
                    eye_startY= 0 if eye_startY<0 else eye_startY
                    #img=img[eye_startY: eye_startY+eye_height+10, eye_startX: eye_startX+ eye_width]
                    img=img[eye_startY+eye_height-17: (eye_startY)+(eye_height+8), eye_startX: eye_startX+ eye_width]
                    cv2.imwrite("./cropped_eye/cropped_eye.png", img)
                        
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
            
            #Function to crop the Forehead Part
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
                    imgsc=imgsc[eye_startY+eye_height+17: eye_startY+eye_height+42, eye_startX-20: eye_startX + 15]
                    cv2.imwrite("./cropped_cheek/cropped_cheek.png", imgsc)
                        
                return 1
            
            ################### Conditions for Dark Circles & Wrinkles Prediction ########
            if st.checkbox("Check Dark Circle Prediction"):
                try:
                    # Function call for face 
                    crop_face_image(face_source_dir, dest_dir, mode)
                    
                    # Function Call for eye
                    crop_eye_image(eye_source_dir, eye_dest_dir, mode)
                    
                    # Function Call for forehead 
                    crop_forehead_image(forehead_source_dir, forehead_dest_dir, mode)
                    
                    # Function Call for cheeks
                    crop_cheek_image(cheek_source_dir, cheek_dest_dir, mode)
                         
                    ######### Dark Circle Prediction ########### Beolw Eye Part
                    img_drk_crkl = cv2.imread('./cropped_eye/cropped_eye.png')
                    img_drk_crkl = cv2.cvtColor(img_drk_crkl, cv2.COLOR_BGR2RGB)
                    img_drk_crkl = cv2.GaussianBlur(img_drk_crkl, (5,5), 0)
                    
                    img_drk_crkl_g = cv2.cvtColor(img_drk_crkl, cv2.COLOR_RGB2GRAY)
                    sobely_img_drk_crkl = cv2.Sobel(img_drk_crkl_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    drkC_hy, drkC_wy = sobely_img_drk_crkl.shape
                    param_img_drk_crkl_1 = cv2.sumElems(sobely_img_drk_crkl)[0]/(drkC_hy * drkC_wy)
                    #st.write("Value of Eye Part is ", param_img_drk_crkl_1)
                    
                    
                    ##Forehead Part
                    img_forehead = cv2.imread('./cropped_forehead/cropped_forehead.png')
                    img_forehead = cv2.cvtColor(img_forehead, cv2.COLOR_BGR2RGB)
                    img_forehead = cv2.GaussianBlur(img_forehead, (5,5), 0)
                    
                    img_forehead_g = cv2.cvtColor(img_forehead, cv2.COLOR_RGB2GRAY)
                    sobely_img_forehead = cv2.Sobel(img_forehead_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    fix_hy, fix_wy = sobely_img_forehead.shape
                    param_img_forehead_1 = cv2.sumElems(sobely_img_forehead)[0]/(fix_hy * fix_wy)
                    #st.write("Value of Forehead Part is ", param_img_forehead_1)
                    
                    ##cheeks Part
                    img_cheeks = cv2.imread('./cropped_cheek/cropped_cheek.png')
                    img_cheeks = cv2.cvtColor(img_cheeks, cv2.COLOR_BGR2RGB)
                    img_cheeks = cv2.GaussianBlur(img_cheeks, (5,5), 0)
                    
                    img_cheeks_g = cv2.cvtColor(img_cheeks, cv2.COLOR_RGB2GRAY)
                    sobely_img_cheeks = cv2.Sobel(img_cheeks_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    chk_hy, chk_wy = sobely_img_cheeks.shape
                    param_img_cheeks_1 = cv2.sumElems(sobely_img_cheeks)[0]/(chk_hy * chk_wy)
                    #st.write("Value of Cheeks Part is ", param_img_cheeks_1)
                    
                    avg_skin_value = (param_img_forehead_1 + param_img_cheeks_1)/2
                    #st.write("Average skin value is ", avg_skin_value)
                    
                    if (param_img_drk_crkl_1 - avg_skin_value) < 0:
                        st.write("No Dark Circles")
                    else:
                        if (param_img_drk_crkl_1 - avg_skin_value) >= 0 and (param_img_drk_crkl_1 - avg_skin_value) <= 5:
                            st.write("Dark Circle Percentage : ", "5.6", " %")
                        else:
                            if (param_img_drk_crkl_1 - avg_skin_value) > 5 and (param_img_drk_crkl_1 - avg_skin_value) <= 13:
                                st.write("Dark Circle Percentage : ", "10.5", " %")
                            else:
                                if (param_img_drk_crkl_1 - avg_skin_value) > 13 and (param_img_drk_crkl_1 - avg_skin_value) <= 20:
                                    st.write("Dark Circle Percentage : ", "15.35", " %")
                                else:
                                    new_avg_skin_value = avg_skin_value * 5
                                    if (param_img_drk_crkl_1 - new_avg_skin_value) >= 0 and (param_img_drk_crkl_1 - new_avg_skin_value) <= 10:
                                        st.write("Dark Circle Percentage : ", "5.6", " %")
                                    else:
                                        if (param_img_drk_crkl_1 - new_avg_skin_value) > 10 and (param_img_drk_crkl_1 - new_avg_skin_value) <= 20:
                                            st.write("Dark Circle Percentage : ", "11.23", " %")
                                        else:
                                            dark_circle = abs(param_img_drk_crkl_1 - new_avg_skin_value)
                                            dark_circle_pred = ((dark_circle)/(param_img_drk_crkl_1 + new_avg_skin_value))*100
                                            if dark_circle_pred > 85:
                                                st.write("Dark Circle Percentage : ", "81.67", " %")
                                            else:
                                                st.write("Dark Circle Percentage : ", round(dark_circle_pred, 2), "%")
                    
                except:
                    st.warning("Please, upload another file!")
                         
                    ######### Dark Circle Prediction ########### Beolw Eye Part
            if st.checkbox("Check Wrinkles Prediction"):
                try:        
                    # Function call for face 
                    crop_face_image(face_source_dir, dest_dir, mode)
                    
                    # Function Call for eye
                    crop_eye_image(eye_source_dir, eye_dest_dir, mode)
                    
                    # Function Call for forehead 
                    crop_forehead_image(forehead_source_dir, forehead_dest_dir, mode)
                    
                    # Function Call for cheeks
                    crop_cheek_image(cheek_source_dir, cheek_dest_dir, mode)
                         
                    ## Conditions for Wrinkles ########
                    img_fix = cv2.imread('Young.jpg')
                    img_fix = cv2.cvtColor(img_fix, cv2.COLOR_BGR2RGB)
                    img_fix = cv2.GaussianBlur(img_fix, (5,5), 0)
                    img_fix_g = cv2.cvtColor(img_fix, cv2.COLOR_RGB2GRAY)
                    sobely_fix = cv2.Sobel(img_fix_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    fix_hy, fix_wy = sobely_fix.shape
                    param_fix_1 = cv2.sumElems(sobely_fix)[0]/(fix_hy * fix_wy)
                    sobelx_fix = cv2.Sobel(img_fix_g, cv2.CV_8UC1, 1, 0, ksize=3)
                    fix_hx, fix_wx = sobelx_fix.shape
                    param_fix_2 = cv2.sumElems(sobelx_fix)[0]/(fix_hx * fix_wx)
                    
                    #image Path
                    imagepath = "./cropped_face/cropped_face.png"
                    # load the input image from disk
                    images = cv2.imread(imagepath)
                    img = images
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.GaussianBlur(img, (5,5), 0)
                    img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    
                    #Horizontal
                    sobely = cv2.Sobel(img_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    #cv2.sumElems(sobely)[0]
                    hy, wy = sobely.shape
                    param_1 = cv2.sumElems(sobely)[0]/(hy * wy)
                    
                    #Vertical
                    sobelx = cv2.Sobel(img_g, cv2.CV_8UC1, 1, 0, ksize=3)
                    #cv2.sumElems(sobelx)[0]
                    hx, wx = sobelx.shape
                    param_2 = cv2.sumElems(sobelx)[0]/(hx * wx)
                    
                    Param_horizontal = (param_1 - param_fix_1)
                    Param_vertical = (param_2 - param_fix_2)
                    #print(Param_horizontal, Param_vertical)
                    #st.write(param_1, param_fix_1, param_2, param_fix_2)
                    
                    if (Param_horizontal) < 0 and (Param_vertical) < 0:
                        st.write("Percentage Wrinkles on Face : " , "5.6 %")
                    else:               
                        if round(((param_1 - param_fix_1)*2),2) >= 80 or round(((param_2 - param_fix_2)*2),2) >=80:
                            st.write("Percentage Wrinkles on Face : " , "82.47 %")
                        else:
                            if Param_horizontal > Param_vertical:
                                st.write("Percentage Wrinkles on Face : ", round(((param_1 - param_fix_1)*3.75),2), "%")
                            else:
                                st.write("Percentage Wrinkles on Face : ", round(((param_2 - param_fix_2)*3.75),2), "%")
                    
                #Show Face of Input Image
                except:
                    st.warning("Please, upload another file!")
        except:
            st.warning("Please, upload another file!")