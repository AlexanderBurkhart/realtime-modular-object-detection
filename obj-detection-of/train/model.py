import numpy as np
import os

import cv2

from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras import losses, optimizers, regularizers

img_rows = 128
img_cols = 256

def get_model():
    print('Implementing model...')
    print('-')
    num_classes = 2	
	
    model = Sequential()	

    model.add(Conv2D(128, (3, 3), input_shape=(256, 128, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(2,2))
	
    Dropout(0.8)
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(2,2))
	
    Dropout(0.8)
    model.add(Flatten())

    model.add(Dense(8, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(num_classes, activation='softmax'))
	
    loss = losses.categorical_crossentropy
    optimizer = optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    print('Done implementing')
    print()

    return model

def preprocess(imgs):
    pimg = []
    for img in imgs:
        resized = cv2.resize(img, (img_rows, img_cols))
        pimg.append(resized/255.)
    return pimg
