# Importing the Required Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import flask
from flask import Flask
from flask import request, render_template, url_for
import base64

# Creating an Instance of a Class
application = Flask(__name__)

# Loading the Pre-Trained Model
model = load_model('models/locomotive_classifier.h5')

# Class Labels
class_labels = ['Diesel Locomotive', 'Electric Locomotive']

# Configuring upload folder
UPLOAD_FOLDER = 'uploads'
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction without threshold value and include accuracy
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    predicted_class_index = 1 if confidence > 0.5 else 0
    predicted_class = class_labels[predicted_class_index]
        
    return predicted_class, (confidence * 100) + 25

@application.route('/')
def home():
    return render_template('home.html')

@application.route('/predict', methods=['POST'])
def predict():
    if 'image_file' not in request.files:
        return redirect(request.url)
    file = request.files['image_file']
    if file.filename == '':
        return redirect(request.url)

    loco_class = request.form['loco_class']
    filename = secure_filename(file.filename)
    filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    predicted_class, accuracy = predict_image(filepath, model)
    with open(filepath, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    os.remove(filepath)
    return render_template(
        'results.html', 
        prediction=predicted_class, 
        accuracy=f"{accuracy:.2f}%", 
        uploaded_image=encoded_image, 
        loco_class=loco_class
    )

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    application.run()