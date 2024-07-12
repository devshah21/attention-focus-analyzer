import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

def classify_image(image_path, model):
    # Load the image
    img = image.load_img(image_path, target_size=(128, 128))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array /= 255.
    
    # Use the model to make a prediction
    prediction = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)
    
    if predicted_class == 0:
        prediction = 'closed eyes'
    else:
        prediction = 'open eyes'
    
    return prediction

model = load_model('eye_Model.h5')

print(classify_image('/Users/devshah/Documents/WorkSpace/CS_Projects/Eye_Concentration/eye-concentration/other_files/_0.jpg', model))

