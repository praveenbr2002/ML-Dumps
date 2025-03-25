import cv2
import numpy as np
#from main_app import *
import matplotlib.pyplot as plt

#Beolw Eye Part
img_drk_crkl = cv2.imread('./cropped_eye/cropped_eye.png')
img_drk_crkl = cv2.cvtColor(img_drk_crkl, cv2.COLOR_BGR2RGB)
img_drk_crkl = cv2.GaussianBlur(img_drk_crkl, (5,5), 0)

img_drk_crkl_g = cv2.cvtColor(img_drk_crkl, cv2.COLOR_RGB2GRAY)
sobely_img_drk_crkl = cv2.Sobel(img_drk_crkl_g, cv2.CV_8UC1, 0, 1, ksize=3)
drkC_hy, drkC_wy = sobely_img_drk_crkl.shape
param_img_drk_crkl_1 = cv2.sumElems(sobely_img_drk_crkl)[0]/(drkC_hy * drkC_wy)
print("Value of Eye Part is ", param_img_drk_crkl_1)


##Forehead Part
img_forehead = cv2.imread('./cropped_forehead/cropped_forehead.png')
img_forehead = cv2.cvtColor(img_forehead, cv2.COLOR_BGR2RGB)
img_forehead = cv2.GaussianBlur(img_forehead, (5,5), 0)

img_forehead_g = cv2.cvtColor(img_forehead, cv2.COLOR_RGB2GRAY)
sobely_img_forehead = cv2.Sobel(img_forehead_g, cv2.CV_8UC1, 0, 1, ksize=3)
fix_hy, fix_wy = sobely_img_forehead.shape
param_img_forehead_1 = cv2.sumElems(sobely_img_forehead)[0]/(fix_hy * fix_wy)
print("Value of Forehead Part is ", param_img_forehead_1)

##cheeks Part
img_cheeks = cv2.imread('./cropped_cheek/cropped_cheek.png')
img_cheeks = cv2.cvtColor(img_cheeks, cv2.COLOR_BGR2RGB)
img_cheeks = cv2.GaussianBlur(img_cheeks, (5,5), 0)

img_cheeks_g = cv2.cvtColor(img_cheeks, cv2.COLOR_RGB2GRAY)
sobely_img_cheeks = cv2.Sobel(img_cheeks_g, cv2.CV_8UC1, 0, 1, ksize=3)
chk_hy, chk_wy = sobely_img_cheeks.shape
param_img_cheeks_1 = cv2.sumElems(sobely_img_cheeks)[0]/(chk_hy * chk_wy)
print("Value of Cheeks Part is ", param_img_cheeks_1)

avg_skin_value = (param_img_forehead_1 + param_img_cheeks_1)/2
print("Average skin value is ", avg_skin_value)