from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from classifier import *
import time
from collections import deque
import numpy as np
import os


N = 60  # Number of seconds to consider
array = deque(maxlen=N)

model = tf.keras.models.load_model('eye_Model.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')





def eye_classifer(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # detects eyes within the detected face area (roi)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # check if any eyes are detected
        if len(eyes) > 0:
            # get the bounding box of the first detected eye
            (ex, ey, ew, eh) = eyes[0]
            
            # crop the region containing the eye
            eye_roi = img[y+ey:y+ey+eh, x+ex:x+ex+ew]
            
            # return the cropped eye image
            return eye_roi
    
    


def is_focused():
    # Calculate the proportion of 'closed eyes' predictions in the last N seconds
    closed_eyes_proportion = np.mean([pred == 'closed eyes' for pred in array])

    # Define a threshold for when the user is considered not focused
    # For example, if the eyes are closed more than 50% of the time
    threshold = 0.5

    return closed_eyes_proportion < threshold


def classify_frame(frame, model):
    # Convert the frame to a numpy array and normalize it
    frame = cv2.resize(frame, (128, 128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.

    # Use the model to make a prediction
    prediction = model.predict(frame)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        prediction = 'closed eyes'
    else:
        prediction = 'open eyes'

    return prediction

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # Use 0 for web camera

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    """Video streaming generator function."""
    frame_count = 0
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            prediction = classify_frame(frame, model)
            array.append(prediction)
            print(array)
            
            if not is_focused():
                print("User is not focused!")
                
            cv2.imwrite(f'test_images/frame_{frame_count}.jpg', frame)
            frame_count += 1
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            time.sleep(2)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)