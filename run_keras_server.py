# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@img.jpg 'http://localhost:5000/predict'

import warnings
warnings.filterwarnings("ignore")

# import the necessary packages
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import tensorflow as tf
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = load_model('main_model.h5')
#fix for an issue with Keras/flask 
graph = tf.get_default_graph()
model._make_predict_function()

class_names = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

def prepare_image(image, target):
	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)

	image = np.expand_dims(image, axis=0)

	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image)) #.convert('L')

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(28, 28))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			with graph.as_default():
				preds = model.predict(image)

			data["predictions"] = []
			data["confidence"] = []

			#make prediction
			data["predictions"].append(class_names[np.argmax(preds)])

			#add confidence
			data["confidence"].append(str(max(100.*preds[0])) + "%")

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run()