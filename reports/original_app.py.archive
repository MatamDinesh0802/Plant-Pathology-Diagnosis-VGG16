from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#import serial

app = Flask(__name__)
MODEL_PATH = 'model_inception.h5'
#MODEL_PATH = 'final_model.h5'

model = load_model(MODEL_PATH)
'''ser = serial.Serial(
    port='COM3',  # Device name
    baudrate=9600,  # Baud rate such as 9600 or 115200 etc.
    parity=serial.PARITY_NONE,  # Enable parity checking
    stopbits=serial.STOPBITS_ONE,  # Number of stop bits
    bytesize=serial.EIGHTBITS,  # Number of data bits.
    timeout=.1,  # Set a read timeout value.
    rtscts=0  # Enable hardware (RTS/CTS) flow control.
)'''

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Bacterialblight"
    elif preds==1:
        preds = "Blast"
    elif preds==2:
        preds = "Brown Spot"
    elif preds==3:
        preds = "Tungro"


    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5003, debug=True)
