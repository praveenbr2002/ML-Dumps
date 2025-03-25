import os
import numpy as np
import base64
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import *
 

app = Flask(__name__)  

 
@app.route('/')  
def upload():  
    return render_template("upload.html")  
"""
# Render the pics
def render_picture(data):
    
    render_pic = base64.b64encode(data).decode('ascii') 
    return render_pic
"""
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        data = f.read()
        
        images = image.img_to_array(data)
        print(type(images))
        
        f.save(f.filename)
        return render_template("success.html", name = f.filename)#, val1 = output1)  
  
if __name__ == '__main__':  
    app.run(debug = True)