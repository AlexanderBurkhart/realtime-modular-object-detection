import numpy as np
import argparse
import os
import glob
import csv
import re

import cv2

import keras
import keras.utils
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

from model import get_model, preprocess

img_rows = 480
img_cols = 640

def train_and_predict():
    
    print('Creating model...')

    model = get_model(int(img_rows/5), int(img_cols/5))
    
    print('Done')
    print('-')

    print('Loading and preprocessing data...')

    imgs_train, imgs_train_mask = load_data('data/train')
    
    print('preprocessing...')
    imgs_train = preprocess(imgs_train)
    imgs_train_mask = preprocess(imgs_train_mask)

    imgs_train = imgs_train.astype('float32')

    imgs_train_mask = imgs_train_mask.astype('float32')
    
    print('Done.')
    print('-')

    print('Training model...')
    print('-')

    model.fit(imgs_train, imgs_train_mask, batch_size=16, epochs=50, verbose=1, shuffle=True, validation_split=0.2)
    
    print('Done')
    print('-')

    print('Saving model as model.h5')

    model.save('model.h5')

    print('Done saving.')

def get_image_and_mask(image_name):
    image_mask_path = image_name.split('.')[0] + '_mask.jpg'
    img = adjust_gamma(cv2.imread(image_name), gamma=0.4)
    img_mask = np.array([])
    
    if os.path.exists(image_mask_path):
    	img_mask = cv2.imread(image_mask_path)
#   else:
#   	print('COULD NOT FIND MASK IMAGE')

    return np.array([img]), np.array([img_mask])

#flip image
def flip(img, dir):
    return cv2.flip(img, dir)

#blur the image with gaussian blur
def blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

#change the img to be brighter
def brightness(img, value=60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


def load_data(folder):
    inc = 6
    total = len(glob.glob(folder+'/*jpg'))*inc//2

    imgs = np.ndarray((total, img_rows, img_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, img_rows, img_cols, 3), dtype=np.uint8)

    print('Creating training images...')
    print('-')

    i = 0
    train_path = os.path.join(folder, '*.jpg')
    images = sorted(glob.glob(train_path))
    for path in images: 
        if 'mask' in path.split('\\')[-1]:
            continue
        #print(path)
        imgs[i], imgs_mask[i] = get_image_and_mask(path)

        imgs[i+1] = adjust_gamma(imgs[i], gamma=0.6) 
        imgs_mask[i+1] = imgs_mask[i]

        imgs[i+2] = blur(imgs[i]) 
        imgs_mask[i+2] = imgs_mask[i]

        imgs[i+3] = blur(imgs[i+1])
        imgs_mask[i+3] = imgs_mask[i]

        imgs[i+4] = flip(imgs[i], 0)
        imgs_mask[i+4] = flip(imgs_mask[i], 0)

        imgs[i+5] = flip(imgs[i], 1)
        imgs_mask[i+5] = flip(imgs_mask[i], 1)

        #cv2.imshow('img', imgs[i+4])
        #cv2.imshow('img_mask', imgs_mask[i+4])
        #cv2.waitKey(0)

        print('Loading frame %i...' % (i/inc))
        i += inc			

    print('Loading done.')
    print('-')	

    print('Returning data...')
    return imgs, imgs_mask

train_and_predict()
