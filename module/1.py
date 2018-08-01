import numpy as np 
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class model:
	@staticmethod
	def build(width=None,height=None,depth=3,classes):
		model=Sequential()
		inputShape=(height,width,depth)

		model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=-1))
		model.add(MaxPooling2D(pool_size=(3,3)))
		model.add(Dropout(0.20))

		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=-1))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=-1))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.20))

		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=-1))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=-1))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(GlobalMaxPooling1D())
		model.add(Dropout(0.20))

		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Dense(classes))
		model.add(Activation('softmax'))

		return model

