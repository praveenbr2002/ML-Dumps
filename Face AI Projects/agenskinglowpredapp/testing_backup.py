# import the necessary packages
import cv2
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

# define the list of age buckets our age detector will predict
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-23)", "(25-35)",
	"(38-43)", "(48-53)", "(60-100)"]

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
            
            ################ Age Prediction ################
            if st.checkbox("Age Prediction"):
                try:
                    # load the input image and construct an input blob for the image
                    images = img_array#cv2.imread(args["image"])
                    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                    results = detector.detect_faces(images)
                    confi = results[0]
                    confidence = confi["confidence"]
                    #st.write("Confi", confidence)
                    st.write("Predicting age of human..!")
                    
                		# compute the (x, y)-coordinates of the bounding box for the
                		# object
                    for person in results:
                        bounding_box = person['box']
                        startX = bounding_box[0]
                        startY = bounding_box[1]
                        endX = bounding_box[0] + bounding_box[2]
                        endY = bounding_box[1] + bounding_box[3]
                        cv2.rectangle(images, (startX, startY), (endX, endY), (0, 255 ,0), 2)
            
                        face = images[startY:endY, startX:endX]
                        #extract the ROI of the face and then construct a blob from *only* the face ROI
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                         (78.4263377603, 87.7689143744, 114.895847746), 
                                                         swapRB=False)
                    if confidence > 0.95:
                    # make predictions on the age and find the age bucket with
                    # the largest corresponding probability
                        ageNet.setInput(faceBlob)
                        preds = ageNet.forward()
                        i = preds[0].argmax()
                        age = AGE_BUCKETS[i]
                		#ageConfidence = preds[0][i]
                		# display the predicted age to our terminal
                        text = "{} Years".format(age)
                        st.write("Age Range : ", age, "Years")
            		    #draw the bounding box of the face along with the associated
            		    # predicted age
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(images, (startX, startY), (endX, endY),
                                      (0, 255, 0), 2)
                        cv2.putText(images, text, (startX, y),
            			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

                    #new_img = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
                    st.image(images, use_column_width=True)
                                
           
                except:
                    st.warning("Please, upload another file!")

        except:
            st.warning("Please, upload another file!")
