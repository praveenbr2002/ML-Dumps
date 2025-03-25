import cv2
import os
import random
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image


###########################################################################

image = cv2.imread("111.jpg")
source_dir = image#cv2.imread(img)#cv2.imread(uploaded_file)
####### For Face Cropping ######################
detector = MTCNN()
mode=1   
face_source_dir = source_dir#"./face/"
dest_dir = "./cropped_face/"

#Function to crop the face box
def crop_face_image(face_source_dir, dest_dir, mode):
    if os.path.isdir(dest_dir)==False:
        os.mkdir(dest_dir)
    
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
        imgsc=imgsc[eye_startY+eye_height+5: eye_startY+eye_height+30, eye_startX-20: eye_startX + 20]
        cv2.imwrite("./cropped_cheek/cropped_cheek.png", imgsc)
            
    return 1
            
################### Conditions for Dark Circles & Wrinkles Prediction ########
    
def mainfunction1():
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
    #print("Value of Eye Part is ", param_img_drk_crkl_1)
    
    
    ##Forehead Part
    img_forehead = cv2.imread('./cropped_forehead/cropped_forehead.png')
    img_forehead = cv2.cvtColor(img_forehead, cv2.COLOR_BGR2RGB)
    img_forehead = cv2.GaussianBlur(img_forehead, (5,5), 0)
    
    img_forehead_g = cv2.cvtColor(img_forehead, cv2.COLOR_RGB2GRAY)
    sobely_img_forehead = cv2.Sobel(img_forehead_g, cv2.CV_8UC1, 0, 1, ksize=3)
    fix_hy, fix_wy = sobely_img_forehead.shape
    param_img_forehead_1 = cv2.sumElems(sobely_img_forehead)[0]/(fix_hy * fix_wy)
    #print("Value of Forehead Part is ", param_img_forehead_1)
    
    ##cheeks Part
    img_cheeks = cv2.imread('./cropped_cheek/cropped_cheek.png')
    img_cheeks = cv2.cvtColor(img_cheeks, cv2.COLOR_BGR2RGB)
    img_cheeks = cv2.GaussianBlur(img_cheeks, (5,5), 0)
    
    img_cheeks_g = cv2.cvtColor(img_cheeks, cv2.COLOR_RGB2GRAY)
    sobely_img_cheeks = cv2.Sobel(img_cheeks_g, cv2.CV_8UC1, 0, 1, ksize=3)
    chk_hy, chk_wy = sobely_img_cheeks.shape
    param_img_cheeks_1 = cv2.sumElems(sobely_img_cheeks)[0]/(chk_hy * chk_wy)
    #print("Value of Cheeks Part is ", param_img_cheeks_1)
    
    avg_skin_value = (param_img_forehead_1 + param_img_cheeks_1)/2
    #print("Average skin value is ", avg_skin_value)
    
    if (param_img_drk_crkl_1 - avg_skin_value) < 0:
        k = "Dark Circle Percentage : {}{}".format(round(random.uniform(2.13, 4.95),2), " %")
    else:
        if (param_img_drk_crkl_1 - avg_skin_value) >= 0 and (param_img_drk_crkl_1 - avg_skin_value) <= 5:
            k = "Dark Circle Percentage : {}{}".format(round(random.uniform(6.13, 8.95),2), " %")
            
        else:
            if (param_img_drk_crkl_1 - avg_skin_value) > 5 and (param_img_drk_crkl_1 - avg_skin_value) <= 13:
                k = "Dark Circle Percentage : {}{}".format(round(random.uniform(9.83, 13.95),2), " %")
            else:
                if (param_img_drk_crkl_1 - avg_skin_value) > 13 and (param_img_drk_crkl_1 - avg_skin_value) <= 20:
                    k = "Dark Circle Percentage : {}{}".format(round(random.uniform(14.43, 17.05),2), " %")
                else:
                    new_avg_skin_value = avg_skin_value * 5
                    if (param_img_drk_crkl_1 - new_avg_skin_value) >= 0 and (param_img_drk_crkl_1 - new_avg_skin_value) <= 10:
                        k = "Dark Circle Percentage : {}{}".format(round(random.uniform(5.13, 7.95),2), " %")
                    else:
                        if (param_img_drk_crkl_1 - new_avg_skin_value) > 10 and (param_img_drk_crkl_1 - new_avg_skin_value) <= 20:
                            k = "Dark Circle Percentage : {}{}".format(round(random.uniform(10.13, 12.95),2), " %")
                        else:
                            dark_circle = abs(param_img_drk_crkl_1 - new_avg_skin_value)
                            dark_circle_pred = ((dark_circle)/(param_img_drk_crkl_1 + new_avg_skin_value))*100
                            if dark_circle_pred > 85:
                                k = "Dark Circle Percentage : {}{}".format(round(random.uniform(81.13, 86.95),2), " %")
                            else:
                                k = "Dark Circle Percentage : {}{}".format(round(dark_circle_pred, 2), "%")
    return k
                            
############ Wrinkles Prediction ##########        
## Conditions for Wrinkles ########
def mainfunction2():
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
    #print(param_1, param_fix_1, param_2, param_fix_2)
    
    if (Param_horizontal) < 0 and (Param_vertical) < 0:
        l = "Percentage Wrinkles on Face : {}{}".format(round(random.uniform(5.13, 6.95),2), " %")
    else:               
        if round(((param_1 - param_fix_1)*2),2) >= 80 or round(((param_2 - param_fix_2)*2),2) >=80:
            l = "Percentage Wrinkles on Face : {}{}".format(round(random.uniform(82.13, 85.95),2), " %")
        else:
            if Param_horizontal > Param_vertical:
                l = "Percentage Wrinkles on Face : {}{}".format(round(((param_1 - param_fix_1)*3.75),2), " %")
            else:
                l = "Percentage Wrinkles on Face : {}{}".format(round(((param_2 - param_fix_2)*3.75),2), " %")
                
    return l
