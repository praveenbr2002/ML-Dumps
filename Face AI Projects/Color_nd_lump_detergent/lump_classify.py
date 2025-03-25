# import the necessary packages
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# loading the detergent detector model from disk
print("loading detergent detector model")
model = load_model("./detergent_model")


### For image
# load the image
image_orig = cv2.imread("./images/images (7).jpeg")
# load the input image from disk
image = image_orig
orig = image.copy()
(h, w) = image.shape[:2]

img = orig
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img_to_array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

# pass the image to predict the use cases
(bad, good) = model.predict(img)[0]
print(bad)
print(good)

# determine the class label and color bounding boxes
label = "Bad Detergent" if bad > good else "Good Detergent"
color = (0, 0, 255) if label == "Bad Detergent" else (0, 255, 0)

# include the probability in the label
label = "{}: {:.2f}%".format(label, max(bad, good) * 100)

# display the label and bounding box rectangle on the output
cv2.putText(image, label, (20, 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
cv2.rectangle(image, (10, 15), (h-10, w-10), color, 2)

# show the output image
cv2.imshow("Image Window", image)
cv2.waitKey(0)