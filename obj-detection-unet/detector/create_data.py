import numpy as np
import csv
import os
from optparse import OptionParser

import cv2

def create_data(num_frames=4500, set='town'):
    #go through each frame and label it accordingly with the csv
    frame_labels = read_csv('data/data.csv')
    final_labels = []
    #print(frame_labels)
    if set=='town':
        start_idx = 8
        end_idx = 12
    elif set=='custom':
        start_idx = 1
        end_idx = 5
    else:
        raise Exception('Unknown type set. Supported types are town and custom.')

    folder = 'train'
    vid = cv2.VideoCapture('data/test.avi')

    if not os.path.exists('data/train'):
        os.mkdir('data/train')
    if not os.path.exists('data/val'):
        os.mkdir('data/val')

    i = 0
    start = 0
    while True:
        read, frame = vid.read()
        if not read or i==num_frames+1:
            break
        if i > num_frames/2:
            folder = 'val'
        name = 'frame'+str(i)
        cv2.imwrite('data/'+folder+'/'+name+'.jpg', frame)

        mask = np.zeros(frame.shape, dtype=np.uint8)

        print('reading frame %i...' % i)
        h = 1
        for j in range(start,len(frame_labels)):
            frame_label = frame_labels[j]
            if(frame_label[1] != i):
                start = j
                break
            #write boundaries
            bbox = []
            for x in range(start_idx,end_idx):
                bbox.append(int(frame_label[x]))
            cv2.rectangle(mask, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (h,h,h), -1)

            h += 1
        
        print('num objects: %i' % h)
        cv2.imwrite('data/'+folder+'/'+name+'_mask.jpg', mask)
        i += 1

def read_csv(path):
    data = []
    with open(path, 'r') as f:
        r = csv.reader(f)
        for row in r:
            row = list(map(float, row))
            data.append(row)
    return data

parser = OptionParser()
parser.add_option('-d', '--data_set', dest='data_set', help='Type of data set.', default='town')
(options, args) = parser.parse_args()

data_set = options.data_set

create_data(set=data_set)
