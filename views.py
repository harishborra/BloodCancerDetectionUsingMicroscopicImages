"""
Routes and views for the flask application.
"""

from datetime import datetime
from BloodCancerDetection import app,static
import streamlit as st
import os
from flask import request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import sys
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
import tensorflow_hub as hub
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.keras import preprocessing

model=Sequential()
model.add(Conv2D(16,(3,3), activation = 'relu', input_shape=(300,300,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation = 'sigmoid'))
classifier_model = model.load_weights('BloodCancerDetection/static/mySavedModel2.hdf5')


COUNT = 0
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global COUNT
    img = request.files['image']
    img.save('BloodCancerDetection/static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('BloodCancerDetection/static/{}.jpg'.format(COUNT))
    img_arr = cv2.resize(img_arr,(300,300))
    img_arr = cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)
    test_image = preprocessing.image.img_to_array(img_arr)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image,axis = 0)
    class_names = ["Image has Detected CANCER ","The image is NORMAL"]
    predictions = model.predict(test_image)
    scores = tf.nn.relu(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    x = round(scores[0],2)
    
    print("************Predictions**************",predictions)
    print("***********",predictions[0])
    COUNT += 1
    return render_template('prediction.html', data=x)

@app.route('/moreinfo')
def moreInfo():
    return render_template("MoreInfo.html")


@app.route('/about')
def about():
    return render_template("about.html")

