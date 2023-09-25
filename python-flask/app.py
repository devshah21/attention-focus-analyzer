from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load the pre-trained object detection model from TensorFlow Hub
detection_model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")

@app.route('/')
def index():
    return render_template('newindex.html')

def detect_objects(frame):
    detections = detection_model(frame)
    return detections

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to match the model's input size (640x640 in this case)
        frame = cv2.resize(frame, (640, 640))

        # Perform object detection on the frame
        detections = detect_objects(frame)

        # Process and draw bounding boxes on the frame (customize this part)
        # ...

        # Convert the frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
