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

def train_and_predict():
    print('Loading and preprocessing data...')

    data, labels = load_data('data/train/final', 'data/train/labels.csv')
    print('preprocessing...')
    imgs_train = preprocess(data)

    imgs_train = np.array(imgs_train)
    labels_train = np.array(labels)

    categorical_labels_train = to_categorical(labels_train)
    
    data, labels = load_data('data/test/final', 'data/test/labels.csv')

    imgs_test = preprocess(data)

    imgs_test = np.array(imgs_test)
    labels_test = np.array(labels)

    categorical_labels_test = to_categorical(labels_test)

    print('Done.')
    print('-')

    print('Creating model...')

    model = get_model()
    
    print('Done')
    print('-')

    print('Training model...')
    print('-')

    model.fit(imgs_train, categorical_labels_train, batch_size=32, epochs=15, verbose=True, validation_split=0.1, shuffle=True)
    
    print('Done training.')
    print('Score: {}'.format(model.evaluate(imgs_test, categorical_labels_test)))

    print('Saving model as model.h5')

    model.save('model.h5')

    print('Done saving.')

def load_data(img_folder, csv_path):
    img_path_folder = os.path.join(img_folder, '*.jpg')
    imgs = glob.glob(img_path_folder)
    imgs.sort(key=natural_keys)

    data = []
    for i in range(0, len(imgs)):
        print('Loading img %i' % i)
        data.append(cv2.imread(imgs[i]))

    labels = []
    with open(csv_path, 'r') as f:
        r = csv.reader(f)
        for row in r:
            labels.append(int(row[0]))

    return data, labels

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

train_and_predict()
