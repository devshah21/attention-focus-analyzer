import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("model_files/shape_predictor_68_face_landmarks.dat")

img = cv2.imread("other_files/woman.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

for face in faces:
    # Get the landmarks/parts for the face in box d.
    landmarks = predictor(gray, face)

    # Loop through all the points
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

    # Draw rectangle around the face
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the output
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def get_feature(landmarks, feature_indices):
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in feature_indices])

# Example: Get left eye coordinates
left_eye_indices = range(36, 42)
left_eye = get_feature(landmarks, left_eye_indices)

# Example: Get right eye coordinates
right_eye_indices = range(42, 48)
right_eye = get_feature(landmarks, right_eye_indices)

# Print eye coordinates
print("Left Eye Coordinates:", left_eye)
print("Right Eye Coordinates:", right_eye)



