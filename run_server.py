# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 21:04:29 2021

@author: HP
"""

from jinja2.loaders import ModuleLoader
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io

flsk = flask.Flask(__name__)

@flsk.route("/", methods=["POST","GET"])
def index():

  if flask.request.method == "GET":
    return flask.render_template('index.html')
  data = {}

  if flask.request.method == "POST":
    if flask.request.files.get("image"):

      image = flask.request.files["image"].read()
      image = Image.open(io.BytesIO(image))

      image = prepare_image(image, target=(224,224))

      preds = model.predict(image)

      if preds[0,0] > 0.5:
        result = "Normal Image"
      else:
        result = "Abnormal Image"
      
      data["prediction result: "] = result

  return flask.jsonify(data)

def prepare_image(image, target):
    
  if image.mode != "RGB":
    image = image.convert("RGB")

  image = image.resize(target)
  image = img_to_array(image)
  image = np.expand_dims(image, axis = 0)
  image = image.astype('float32')
  image = image / 255

  return image

if __name__ == "main":
  print(("Flask starting server..."
  "please wait until server has fully started"))
  global model
  model = load_model('medical_diagnosis_cnn_model.h5')
  flsk.run()