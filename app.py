# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 21:02:33 2021

@author: HP
"""

from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import streamlit as st
import io

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image.astype('float32')
    image = image / 255

    return image

model = load_model('C:/Users/HP/spyder/Model/medical_diagnosis_cnn_model.h5')

st.header("Convolutional Neural Network for Medical Image Analysis")

file_uploaded = st.file_uploader("Choose the Image File", type=['jpg', 'jpeg', 'png'])
    
if file_uploaded is not None:
    image = Image.open(file_uploaded)
      
    image = prepare_image(image, target=(224,224))


    preds = model.predict(image)

    if preds[0,0] > 0.5:
        result="Normal Image"
    else:
        result = "Abnormal Image"

    col1, col2 = st.columns(2)
    col1.image(image, caption="The image is classified as "+result, width=300)
    col2.header("Classification Result")
    col2.write("The image is classified as "+result)