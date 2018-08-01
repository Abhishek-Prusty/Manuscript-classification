import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from module.vggmodel import modelVGG
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.callbacks import EarlyStopping

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True)
ap.add_argument("-m","--model",required=True)
ap.add_argument("-p","--plot",type=str,default="plot.png")
args=vars(ap.parse_args())

EPOCHS = 25
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (100, 1000, 3)

data=[]
labels=[]

print("loading images")
imagePaths=sorted(list(paths.list_images(args["dataset"])))
random.seed(10)
random.shuffle(imagePaths)


for imagePath in imagePaths:
	image=cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image=img_to_array(image)
	data.append(image)

	label=imagePath.split(os.path.sep)[-2]
	labels.append(label)

data=np.array(data,dtype="float")/255.0
labels=np.array(labels)

lb=LabelBinarizer()
labels=lb.fit_transform(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.2,random_state=10)

aug=ImageDataGenerator(rotation_range=5,width_shift_range=0.1,
	height_shift_range=0.1,shear_range=0.1,zoom_range=0.2,
	horizontal_flip=True,fill_mode="nearest")

print("compilingg model ")
model=modelVGG.build(IMAGE_DIMS[1],IMAGE_DIMS[0],IMAGE_DIMS[2],classes=len(lb.classes_))

opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,
	metrics=["accuracy"])

stop_here = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

print("training ")
H = model.fit_generator(aug.flow(trainX,trainY,batch_size=BS),
	validation_data=(testX,testY),
	steps_per_epoch=len(trainX)//BS,
	epochs=EPOCHS,verbose=1,
	callbacks=[stop_here])

model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
