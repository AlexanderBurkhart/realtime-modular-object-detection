import numpy as np
from numpy import unique
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
        label_idx = 1
    elif set=='custom':
        start_idx = 1
        end_idx = 5
        label_idx = 0
    else:
        raise Exception('Unknown type set. Supported types are town and custom.')

    folder = 'train'
    vid = cv2.VideoCapture('data/test.avi')

    if not os.path.exists('data/train'):
        os.mkdir('data/train')
    if not os.path.exists('data/val'):
        os.mkdir('data/val')

    i = 0
    name_idx = 0
    start = 0
    while True:
        read, frame = vid.read()
        if not read or i==num_frames+1:
            break
        if i > num_frames/2:
            folder = 'val' 
        
        #create mask
        mask = np.zeros(frame.shape, dtype=np.uint8)
        print('reading frame %i...' % i)
        h = 1
        for j in range(start,len(frame_labels)):
            frame_label = frame_labels[j]
            if(frame_label[label_idx] != i):
                start = j
                break
            #write boundaries
            bbox = []
            for x in range(start_idx,end_idx):
                bbox.append(int(frame_label[x]))
           
            if(len(unique(bbox)) <= 1):
                continue

            #print('tl_x: %i, tl_y: %i, br_x: %i, br_y: %i' % (bbox[0], bbox[1], bbox[2], bbox[3]))
            
            #rectangle mask
            #cv2.rectangle(mask, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,255,255), -1)

            #circle mask
            cv2.circle(mask, (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)), 
                        int((bbox[2]-bbox[0])/2), (255,255,255), -1)

            h += 1

        #cv2.imshow('mask', mask)
        #cv2.imshow('img', frame)
        #cv2.waitKey(0)
        print('num objects: %i' % h)

        #no edit
        cv2.imwrite('data/'+folder+'/frame'+str(name_idx)+'.jpg', frame)
        cv2.imwrite('data/'+folder+'/frame'+str(name_idx)+'_mask.jpg', mask)
        name_idx += 1

        #brighter
        cv2.imwrite('data/'+folder+'/frame'+str(name_idx)+'.jpg', adjust_gamma(frame, gamma=1.5))
        cv2.imwrite('data/'+folder+'/frame'+str(name_idx)+'_mask.jpg', mask)
        name_idx += 1
        
        #darker
        cv2.imwrite('data/'+folder+'/frame'+str(name_idx)+'.jpg', adjust_gamma(frame, gamma=0.5))
        cv2.imwrite('data/'+folder+'/frame'+str(name_idx)+'_mask.jpg', mask)
        name_idx += 1
        
        #blur
        cv2.imwrite('data/'+folder+'/frame'+str(name_idx)+'.jpg', blur(frame, k=7))
        cv2.imwrite('data/'+folder+'/frame'+str(name_idx)+'_mask.jpg', mask)
        name_idx += 1

        i += 1

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    	for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def blur(img, k=0):
    return cv2.GaussianBlur(img, (k,k), 0)

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
