# import the necessary packages
from model import *
from flask import Flask, render_template, request, Response

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('Home_page.html')


@app.route('/res',methods = ['POST','GET'])
def res():
	global result
	if request.method == 'POST':
		result = request.form.to_dict()
		return render_template("results.html",result = result)

@app.route('/results')
def video_feed():
	global result
	params= result
	return Response(pyshine_process(params),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,threaded=True)
    app.run(host='0.0.0.0', port=5000)