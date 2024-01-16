# Eye Focus Detection Web App

This is a Flask web application that uses OpenCV and a trained TensorFlow model to detect whether a user's eyes are open or closed. If the user's eyes are closed for more than a certain proportion of time, the system will alert the user to focus.

## Dependencies

- Flask
- OpenCV
- TensorFlow
- time
- collections
- numpy
- os

## How It Works

1. The application uses a webcam to capture video frames.
2. Each frame is processed to detect faces and eyes using OpenCV's Haar cascades.
3. If eyes are detected, the region of the image containing the eyes is passed to a pre-trained TensorFlow model, which classifies the eyes as either open or closed.
4. The application keeps track of the eye state predictions over the last N seconds. If the proportion of 'closed' predictions exceeds a certain threshold, the system alerts the user to focus.

## Code Structure

- `eye_classifer(img)`: This function takes an image as input, detects faces and eyes in the image using Haar cascades, and returns the region of the image containing the first detected eye.
- `is_focused()`: This function calculates the proportion of 'closed eyes' predictions in the last N seconds and returns whether this proportion is below a certain threshold.
- `classify_frame(frame, model)`: This function takes a frame and a model as input, preprocesses the frame, uses the model to classify the frame as 'open eyes' or 'closed eyes', and returns the prediction.
- `gen()`: This is a generator function that reads frames from the webcam, processes each frame, makes predictions, checks if the user is focused, and yields the processed frames.
- `video_feed()`: This is a route that returns the response from the `gen()` function.
- `index()`: This is the home page route that renders the `index.html` template.

## Running the App

To run the app, execute the script. The app will start on `0.0.0.0` port `8080`.

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
