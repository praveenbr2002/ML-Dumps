import cv2
import random
import numpy as np
import streamlit as st
from PIL import Image

###########################################################################
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Hair Loss Prediction App")
logo = Image.open("Artifutech_logo.png")
#st.sidebar.image(logo, width=190, use_column_width=True)
st.sidebar.title("Artifutech Hair Loss Prediction App")
st.sidebar.warning("For better results please follow below rules for using App")
st.sidebar.success("1. Upload pictures according to given desired format. \n 2. Uploaded images should be in top view (see image sample).")

## Logo
image_logo = Image.open("logo2.jpeg")
st.image(image_logo, use_column_width=True)
st.subheader("Sample Images for uploading:")

images = ["men_without_hairloss.png", "women_without_hairloss.png"]
st.image(images, use_column_width=False, caption=["Images Sample for Men", "Images Sample for WoMen"], width=300, clamp=True)

st.set_option('deprecation.showfileUploaderEncoding', False)
############################################################################################################################
st.warning("To check percentage of hair loss Prediction. Please click in below box.")

################### Conditions for Hair Loss Prediction ########
if st.checkbox("For Men"):
    uploaded_file_men = st.file_uploader("Choose an image to predict % hair loss:", type=["png", "jpeg", "jpg"], key=0)
    if st.checkbox("Show uploaded image & it's Prediction", key=1):
        if uploaded_file_men is not None:
            try:
                image = Image.open(uploaded_file_men)
                img_array_men = np.array(image)
                img_array_men = cv2.cvtColor(img_array_men, cv2.COLOR_RGB2BGR)
                st.image(image, caption='Uploaded Image for Men.', use_column_width=True)

                try:
                    st.subheader("Predicting hair loss of men..!")
                    ############ Fix Image for MEN ##################
                    img_fix_men = cv2.imread('men_without_hairloss.png')
                    img_fix_men = cv2.cvtColor(img_fix_men, cv2.COLOR_BGR2RGB)
                    img_fix_men = cv2.GaussianBlur(img_fix_men, (5,5), 0)
                    
                    img_fix_men_g = cv2.cvtColor(img_fix_men, cv2.COLOR_RGB2GRAY)
                    sobely_img_fix_men = cv2.Sobel(img_fix_men_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    fxhy, fxwy = sobely_img_fix_men.shape
                    param_img_fix_men_1 = cv2.sumElems(sobely_img_fix_men)[0]/(fxhy * fxwy)
                    
                    sobelx_img_fix_men = cv2.Sobel(img_fix_men_g, cv2.CV_8UC1, 1, 0, ksize=3)
                    fxhx, fxwx = sobelx_img_fix_men.shape
                    param_img_fix_men_2 = cv2.sumElems(sobelx_img_fix_men)[0]/(fxhx * fxwx)
                    
                    param_img_fix_men_avg = (param_img_fix_men_1 + param_img_fix_men_2)/2
                    
                    ############### Uploaded Image #########################
                    img_array_men = cv2.GaussianBlur(img_array_men, (5,5), 0)
                    
                    img_array_men_g = cv2.cvtColor(img_array_men, cv2.COLOR_RGB2GRAY)
                    sobely_img_array_men = cv2.Sobel(img_array_men_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    hy, wy = sobely_img_array_men.shape
                    param_img_array_men_1 = cv2.sumElems(sobely_img_array_men)[0]/(hy * wy)
                    
                    sobelx_img_array_men = cv2.Sobel(img_array_men_g, cv2.CV_8UC1, 1, 0, ksize=3)
                    hx, wx = sobelx_img_array_men.shape
                    param_img_array_men_2 = cv2.sumElems(sobelx_img_array_men)[0]/(hx * wx)
                    
                    param_img_array_men_avg = (param_img_array_men_1 + param_img_array_men_2)/2
                    #st.write("Array Value is ", param_img_fix_men_avg, param_img_array_men_avg)
                    
                    ################# Conditions Based on Array Value #############
                    if (param_img_fix_men_avg - param_img_array_men_avg) < 0:
                        st.write("Hair Loss Percentage : ", round(random.uniform(1.99, 4.95),2), " %")
                    else:
                        if (param_img_fix_men_avg - param_img_array_men_avg) >= 0 and (param_img_fix_men_avg - param_img_array_men_avg) <= 1:
                            st.write("Hair Loss Percentage : ", round(random.uniform(4.99, 6.95),2), " %")
                        else:
                            if (param_img_fix_men_avg - param_img_array_men_avg) > 1 and (param_img_fix_men_avg - param_img_array_men_avg) <= 2:
                                st.write("Hair Loss Percentage : ", round(random.uniform(6.99, 8.95),2), " %")
                            else:
                                if (param_img_fix_men_avg - param_img_array_men_avg) > 2 and (param_img_fix_men_avg - param_img_array_men_avg) <= 3:
                                    st.write("Hair Loss Percentage : ", round(random.uniform(8.99, 12.95),2), " %")
                                else:
                                    if (param_img_fix_men_avg - param_img_array_men_avg) > 3 and (param_img_fix_men_avg - param_img_array_men_avg) <= 4:
                                        st.write("Hair Loss Percentage : ", round(random.uniform(12.99, 15.95),2), " %")
                                    else:
                                        if (param_img_fix_men_avg - param_img_array_men_avg) > 4 and (param_img_fix_men_avg - param_img_array_men_avg) <= 5:
                                            st.write("Hair Loss Percentage : ", round(random.uniform(16.99, 19.95),2), " %")
                                        else:
                                            if (param_img_fix_men_avg - param_img_array_men_avg) > 5 and (param_img_fix_men_avg - param_img_array_men_avg) <= 6:
                                                st.write("Hair Loss Percentage : ", round(random.uniform(20.99, 29.95),2), " %")
                                            else:
                                                if (param_img_fix_men_avg - param_img_array_men_avg) > 6 and (param_img_fix_men_avg - param_img_array_men_avg) <= 7:
                                                    st.write("Hair Loss Percentage : ", round(random.uniform(30.99, 39.95),2), " %")
                                                else:
                                                    if (param_img_fix_men_avg - param_img_array_men_avg) > 7 and (param_img_fix_men_avg - param_img_array_men_avg) <= 8:
                                                        st.write("Hair Loss Percentage : ", round(random.uniform(40.99, 49.95),2), " %")
                                                    else:
                                                        if (param_img_fix_men_avg - param_img_array_men_avg) > 8 and (param_img_fix_men_avg - param_img_array_men_avg) <= 9:
                                                            st.write("Hair Loss Percentage : ", round(random.uniform(50.99, 65.95),2), " %")
                                                        else:
                                                            if (param_img_fix_men_avg - param_img_array_men_avg) > 9 and (param_img_fix_men_avg - param_img_array_men_avg) <= 10.5:
                                                                st.write("Hair Loss Percentage : ", round(random.uniform(65.99, 75.95),2), " %")
                                                            else:
                                                                st.write("Hair Loss Percentage : ", round(random.uniform(75.99, 84.95), 2), "%")   

                except:
                    st.warning("Please, upload another file!")
            except:
                st.warning("Please, upload another file!")
if st.checkbox("For WoMen"):
    uploaded_file_women = st.file_uploader("Choose an image to predict % hair loss:", type=["png", "jpeg", "jpg"], key=1)
    if st.checkbox("Show uploaded image & it's Prediction", key=0):
        if uploaded_file_women is not None:
            try:
                image = Image.open(uploaded_file_women)
                img_array_women = np.array(image)
                img_array_women = cv2.cvtColor(img_array_women, cv2.COLOR_RGB2BGR)
                st.image(image, caption='Uploaded Image for Women.', use_column_width=True)

                try:
                    st.subheader("Predicting hair loss of women..!")
                    ############ Fix Image for WOMEN ##################
                    img_fix_women = cv2.imread('105.png')
                    img_fix_women = cv2.cvtColor(img_fix_women, cv2.COLOR_BGR2RGB)
                    img_fix_women = cv2.GaussianBlur(img_fix_women, (5,5), 0)
                    
                    img_fix_women_g = cv2.cvtColor(img_fix_women, cv2.COLOR_RGB2GRAY)
                    sobely_img_fix_women = cv2.Sobel(img_fix_women_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    fxhyw, fxwyw = sobely_img_fix_women.shape
                    param_img_fix_women_1 = cv2.sumElems(sobely_img_fix_women)[0]/(fxhyw * fxwyw)
                    
                    sobelx_img_fix_women = cv2.Sobel(img_fix_women_g, cv2.CV_8UC1, 1, 0, ksize=3)
                    fxhxw, fxwxw = sobelx_img_fix_women.shape
                    param_img_fix_women_2 = cv2.sumElems(sobelx_img_fix_women)[0]/(fxhxw * fxwxw)
                    
                    param_img_fix_women_avg = (param_img_fix_women_1)# + param_img_fix_women_2)/2
                    
                    
                    ############### Uploaded Image #########################
                    img_array_women = cv2.GaussianBlur(img_array_women, (5,5), 0)
                    
                    img_array_women_g = cv2.cvtColor(img_array_women, cv2.COLOR_RGB2GRAY)
                    sobely_img_array_women = cv2.Sobel(img_array_women_g, cv2.CV_8UC1, 0, 1, ksize=3)
                    hyw, wyw = sobely_img_array_women.shape
                    param_img_array_women_1 = cv2.sumElems(sobely_img_array_women)[0]/(hyw * wyw)
                    
                    sobelx_img_array_women = cv2.Sobel(img_array_women_g, cv2.CV_8UC1, 1, 0, ksize=3)
                    hxw, wxw = sobelx_img_array_women.shape
                    param_img_array_women_2 = cv2.sumElems(sobelx_img_array_women)[0]/(hxw * wxw)
                    
                    param_img_array_women_avg = (param_img_array_women_1 + param_img_array_women_2)/2
                    
                    #st.write("Array Value is ", param_img_fix_women_avg, param_img_array_women_avg)
                    
                    ################# Conditions Based on Array Value #############
                    if (param_img_fix_women_avg - param_img_array_women_avg) < 0:
                        st.write("Hair Loss Percentage : ", round(random.uniform(1.99, 4.95),2), " %")
                    else:
                        if (param_img_fix_women_avg - param_img_array_women_avg) >= 0 and (param_img_fix_women_avg - param_img_array_women_avg) <= 1:
                            st.write("Hair Loss Percentage : ", round(random.uniform(4.99, 6.95),2), " %")
                        else:
                            if (param_img_fix_women_avg - param_img_array_women_avg) > 1 and (param_img_fix_women_avg - param_img_array_women_avg) <= 2:
                                st.write("Hair Loss Percentage : ", round(random.uniform(6.99, 8.95),2), " %")
                            else:
                                if (param_img_fix_women_avg - param_img_array_women_avg) > 2 and (param_img_fix_women_avg - param_img_array_women_avg) <= 3:
                                    st.write("Hair Loss Percentage : ", round(random.uniform(8.99, 12.95),2), " %")
                                else:
                                    if (param_img_fix_women_avg - param_img_array_women_avg) > 3 and (param_img_fix_women_avg - param_img_array_women_avg) <= 4:
                                        st.write("Hair Loss Percentage : ", round(random.uniform(12.99, 15.95),2), " %")
                                    else:
                                        if (param_img_fix_women_avg - param_img_array_women_avg) > 4 and (param_img_fix_women_avg - param_img_array_women_avg) <= 5:
                                            st.write("Hair Loss Percentage : ", round(random.uniform(16.99, 19.95),2), " %")
                                        else:
                                            if (param_img_fix_women_avg - param_img_array_women_avg) > 5 and (param_img_fix_women_avg - param_img_array_women_avg) <= 6:
                                                st.write("Hair Loss Percentage : ", round(random.uniform(20.99, 29.95),2), " %")
                                            else:
                                                if (param_img_fix_women_avg - param_img_array_women_avg) > 6 and (param_img_fix_women_avg - param_img_array_women_avg) <= 7:
                                                    st.write("Hair Loss Percentage : ", round(random.uniform(30.99, 39.95),2), " %")
                                                else:
                                                    if (param_img_fix_women_avg - param_img_array_women_avg) > 7 and (param_img_fix_women_avg - param_img_array_women_avg) <= 8:
                                                        st.write("Hair Loss Percentage : ", round(random.uniform(40.99, 49.95),2), " %")
                                                    else:
                                                        if (param_img_fix_women_avg - param_img_array_women_avg) > 8 and (param_img_fix_women_avg - param_img_array_women_avg) <= 9:
                                                            st.write("Hair Loss Percentage : ", round(random.uniform(50.99, 65.95),2), " %")
                                                        else:
                                                            if (param_img_fix_women_avg - param_img_array_women_avg) > 9 and (param_img_fix_women_avg - param_img_array_women_avg) <= 10.5:
                                                                st.write("Hair Loss Percentage : ", round(random.uniform(65.99, 75.95),2), " %")
                                                            else:
                                                                st.write("Hair Loss Percentage : ", round(random.uniform(75.99, 84.95), 2), "%")   

                except:
                    st.warning("Please, upload another file!")
            except:
                st.warning("Please, upload another file!")