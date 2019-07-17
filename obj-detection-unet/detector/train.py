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

img_rows = 1080
img_cols = 1920

def train_and_predict():
    print('Loading and preprocessing data...')

    imgs_train, imgs_train_mask = load_data('data/train')
    
    print('preprocessing...')
    imgs_train = preprocess(imgs_train)
    imgs_train_mask = preprocess(imgs_train_mask)

    imgs_train = imgs_train.astype('float32')

    imgs_train_mask = imgs_train_mask.astype('float32')
    
    print('Done.')
    print('-')

    print('Creating model...')

    model = get_model()
    
    print('Done')
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
    img = cv2.imread(image_name)
    img_mask = np.array([])
    
    if os.path.exists(image_mask_path):
    	img_mask = cv2.imread(image_mask_path)
#   else:
#   	print('COULD NOT FIND MASK IMAGE')

    return np.array([img]), np.array([img_mask])

#blur the image with gaussian blur
def blur(img, k=3):
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

def load_data(folder):
    inc = 4
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

        imgs[i+1] = brightness(imgs[i])
        imgs_mask[i+1] = imgs_mask[i]

        imgs[i+2] = blur(imgs[i])
        imgs_mask[i+2] = imgs_mask[i]

        imgs[i+3] = blur(imgs[i+1])
        imgs_mask[i+3] = imgs_mask[i]

        print('Loading frame %i...' % (i/inc))
        i += inc			

    print('Loading done.')
    print('-')	

    print('Returning data...')
    return imgs, imgs_mask

#def load_data(datapath):
#    imgs = []
#    bboxes = []
#    csvpath = datapath+'train.csv'
#   
#    labels = []
#    with open(csvpath, 'r') as f:
#        prev_imgpath = ''
#        frame_bboxes = []
#        r = csv.reader(f)
#        for row in r:
#            if prev_imgpath=='':
#                prev_imgpath = row[0]
#                img = cv2.imread(row[0])
#                
#            if not prev_imgpath==row[0]:
#                print('Grabbed data from frame: ' + prev_imgpath)
#                imgs.append(img)
#                bboxes.append(frame_bboxes)
#                return imgs, bboxes
#                frame_bboxes = []
#                img = cv2.imread(row[0])
#                prev_imgpath = row[0]
#            
#            bbox = []
#            for i in range(1,5):
#                bbox.append(int(row[i]))
#            frame_bboxes.append(bbox)
#        
#        return imgs, bboxes

train_and_predict()
