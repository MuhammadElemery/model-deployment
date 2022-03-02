from cv2 import CV_8UC3
from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np


MODEL_PATH = r'brac.h5'
model = keras.models.load_model(MODEL_PATH)


app = Flask(__name__)
app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main():
	return "home"

@app.route('/upload', methods=['POST'])
def upload_file():
	if 'files[]' not in request.files:
		resp = jsonify({'message':'No File part in the request'})
		resp.status_code =400
		return resp

	files = request.files.getlist('files[]')
	
	errors = {}
	success = False
	
	for file in files:
		
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			success = True
		else:
			errors[file.filename] = 'File type is not allowed'
 
	if success and errors:
		errors['message'] = 'File(s) successfully uploaded'
		resp = jsonify(errors)
		resp.status_code = 500
		return resp
	if success:
			resp = jsonify({'message' : 'Files successfully uploaded'})
			resp.status_code = 201

			image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			dim = (128,128)
			re_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
			re_image = re_image.reshape(1, re_image.shape[0], re_image.shape[1], 1)

			x =model.predict(re_image)
			x = x.reshape(128, 128,1)
			dim = (600, 600)
  
			# resize image
			resized = cv2.resize(x, dim, interpolation = cv2.INTER_AREA)
			masked_path = 'static/masked/' + 'masked' + filename
			frame_normed = 255 * (resized - resized.min()) / (resized.max() - resized.min())
			frame_normed = np.array(frame_normed, np.int)
			cv2.imwrite(masked_path, frame_normed)

			return resp
	else:
		resp = jsonify(errors)
		resp.status_code = 500
		return resp
	


if __name__ == '__main__':
	app.run(port = 5053,debug = False)