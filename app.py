from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import torch
import torchvision.transforms as transforms
import time
from collections import deque
import numpy as np
import os
from os import system
import logging

from eye_detection import EyeDetectionCNN  

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Global variables
N = 45  # Number of seconds to consider
array = deque(maxlen=N)
threshold = None
camera = None
model = None

# load model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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
    
    return None

def is_focused():
    # Calculate the proportion of 'closed eyes' predictions in the last N seconds
    closed_eyes_proportion = np.mean([pred == 'closed eyes' for pred in array])

    # Use the user-defined threshold
    return closed_eyes_proportion < threshold

def classify_frame(frame, model):
    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    frame = frame.unsqueeze(0)  # Add batch dimension

    # Use the model to make a prediction
    with torch.no_grad():
        output = model(frame)
        _, predicted = torch.max(output, 1)

    if predicted.item() == 0:
        prediction = 'closed eyes'
    else:
        prediction = 'open eyes'

    return prediction

@app.route('/')
def index():
    """Threshold input page."""
    app.logger.info("Accessing index page")
    return render_template('threshold.html')

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global threshold
    app.logger.info("Setting threshold")
    threshold = float(request.form['threshold'])
    app.logger.info(f"Threshold set to: {threshold}")
    return redirect(url_for('video_page'))

@app.route('/video')
def video_page():
    """Video streaming page."""
    app.logger.info("Accessing video page")
    if threshold is None:
        app.logger.warning("Threshold not set, redirecting to index")
        return redirect(url_for('index'))
    return render_template('video.html')

def gen():
    """Video streaming generator function."""
    global camera, model
    
    # Initialize camera and model here
    if camera is None:
        camera = cv2.VideoCapture(0)
    if model is None:
        model = EyeDetectionCNN()
        model.load_state_dict(torch.load('model_files/eye_detection_cnn.pth'))
        model.eval()
    
    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            
            # Use the eye_classifier function to crop the eyes from the frame
            eye_roi = eye_classifer(frame)
            
            # If eyes are detected in the frame
            if eye_roi is not None:
                # Classify the cropped eyes
                prediction = classify_frame(eye_roi, model)
                array.append(prediction)
                print(array)
                
                if not is_focused():
                    print("User is not focused!")
                    system('say FOCUS UP, YOU ARE GETTING DISTRACTED')
                    
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
    app.logger.info("Starting the application")
    app.run(host='0.0.0.0', port=8080, debug=True)