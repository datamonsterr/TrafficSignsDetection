import keras
import cv2
from imageio.v2 import imread
import os
import numpy as np

# disable using GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model("traffic_sign_model.h5")


input_img = cv2.resize(imread("bienbao102-0x0.jpg"), (64, 64))
refined_img = np.reshape(input_img, (0, 64, 64, 3))
prediction = model.predict(refined_img)


# open camera
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=2.0, fy=2.0,
                       interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    cv2.rectangle(frame, (1, 1), (1, 1), (255, 255, 255), 2)
    # Hit escape to exit
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
