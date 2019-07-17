import numpy as np
import csv

import cv2

def create_data():
    #go through each frame and label it accordingly with the csv
    frame_labels = read_csv('data/data.csv')
    final_labels = []
    #print(frame_labels)
   
    folder = 'train'
    vid = cv2.VideoCapture('data/test.avi')
    i = 0
    start = 0
    while True:
        read, frame = vid.read()
        if not read or i==4500+1:
            break
        if i > 2250:
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
            for x in range(8,12):
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

create_data()
