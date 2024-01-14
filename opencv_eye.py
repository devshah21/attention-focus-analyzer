import os
import cv2

# read input image
img = cv2.imread('woman.jpg')

# convert to grayscale of each frame
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# read the haarcascade to detect the faces in an image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# read the haarcascade to detect the eyes in an image
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# detects faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.3, 4)
print('Number of detected faces:', len(faces))

# create a directory to store the cropped images
output_directory = 'cropped_images'
os.makedirs(output_directory, exist_ok=True)

# loop over the detected faces
for i, (x, y, w, h) in enumerate(faces):
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # detects eyes within the detected face area (roi)
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # loop over the detected eyes
    for j, (ex, ey, ew, eh) in enumerate(eyes):
        # crop the region containing the eyes
        eye_roi = roi_color[ey:ey+eh, ex:ex+ew]

        # save the cropped eye image
        output_path = os.path.join(output_directory, f"face_{i}_eye_{j}.png")
        cv2.imwrite(output_path, eye_roi)

        # draw a rectangle around eyes
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

# display the image with detected eyes
