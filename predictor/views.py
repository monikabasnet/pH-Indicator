from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
import numpy as np
import joblib
import os


# loading the InceptionV# model for feature extraction without top layer

base_model =  InceptionV3(weights = 'imagenet', include_top = False , pooling = 'avg')
model_path = os.path.join(os.path.dirname(__file__), 'model/ph_predictor_rf_model.pkl')
rf_model = joblib.load(model_path)


def extract_features(img_path):
    # Resizing the Incepion V3
    img = keras_image.load_img(img_path , target_size = (299, 299))
    # Converting to numpy array
    img_array = keras_image.img_to_array(img)
    #Adding batch dimension
    img_array = np.expand_dims(img_array, axis = 0)
    #Noramlize for Inception V3
    img_array = preprocess_input(img_array)
    #Extracting features
    features = base_model.predict(img_array)
    return features.flatten()


def upload_image(request):
    if request.method == 'POST':
        uploaded_image = request.FILES.get('image')
        if uploaded_image:
            fs = FileSystemStorage()
            filename = fs.save(uploaded_image.name, uploaded_image)
            image_path = fs.path(filename)
            image_url = fs.url(filename)

            # Extract features & predict
            features = extract_features(image_path).reshape(1, -1)
            predicted_ph = rf_model.predict(features)[0]

            return render(request, 'predictor/result.html', {
                'image_url': image_url,
                'predicted_ph': round(predicted_ph, 2)
            })

    return render(request, 'predictor/upload.html')
