from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")

args = vars(ap.parse_args())
image = cv2.imread(args["image"])
image = cv2.resize(image, (1000, 100))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model(args["model"])
proba = model.predict(image)[0]
print(proba)
if(proba>0.5):
	print("khmer")
else:
	print("balinese")
