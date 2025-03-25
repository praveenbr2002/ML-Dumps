import cv2
import numpy as np
from main_app import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

#Loading Model
model  = load_model("model_new.h5")

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
imagepath = "./cropped_img/cropped_raw_img.png"

# load the input image from disk
images = cv2.imread(imagepath)

orig = images.copy()
(h, w) = images.shape[:2]

face = orig
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
face = cv2.resize(face, (224, 224))
face = img_to_array(face)
face = preprocess_input(face)
face = np.expand_dims(face, axis=0)

(Adults, Children, MiddleAges, Old, Youth) = model.predict(face)[0]
#print(model.predict(face)[0])

if (Children < Adults) and (Children < MiddleAges) and (Children < Old) and (Children < Youth):
    label = "0-14"
elif (Youth > Children) and (Youth < Adults) and (Youth < MiddleAges) and (Youth < Old):
    label = "15-24"
elif (Adults > Children) and (Adults > Youth) and (Adults < MiddleAges) and (Adults < Old):
    label = "25-39"
elif (MiddleAges > Children) and (MiddleAges > Adults) and (MiddleAges > Youth) and (MiddleAges < Old):
    label = "40-59"
else:
    label="60+"

color = (255, 0, 0)
label = "Age Group:{}".format(label)#, max(Adults, Children, MiddleAges, Old, Youth) * 100)

print(label)

img = orig
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.GaussianBlur(img, (5,5), 0)
img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img)
plt.show()

sobely = cv2.Sobel(img_g, cv2.CV_8UC1, 0, 1, ksize=3)
#plt.imshow(sobely,  cmap='gray')
#plt.show()
cv2.sumElems(sobely)[0]
hy, wy = sobely.shape
param_1 = cv2.sumElems(sobely)[0]/(hy * wy)

sobelx = cv2.Sobel(img_g, cv2.CV_8UC1, 1, 0, ksize=3)
cv2.sumElems(sobelx)[0]
hx, wx = sobelx.shape
param_2 = cv2.sumElems(sobelx)[0]/(hx * wx)

Param_horizontal = (param_1 - param_fix_1)
Param_vertical = (param_2 - param_fix_2)
#print(Param_horizontal, Param_vertical)
#print(param_1, param_fix_1, param_2, param_fix_2)

if label == "0-14":
    print("No Wrinkles")
    
else:
    if (Param_horizontal) < 0 and (Param_vertical) < 0:
        print("No Wrinkles")
    else:
        if Param_horizontal > Param_vertical:
            print("Horizontal", (param_1 - param_fix_1)*2, "%")
        else:
            print("Vertical", (param_2 - param_fix_2)*2, "%")
    
