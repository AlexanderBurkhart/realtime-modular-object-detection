import numpy as np
import csv

import cv2

def create_data():
    #go through each frame and label it accordingly with the csv
    frame_labels = read_csv('data/data.csv')
    final_labels = []
    #print(frame_labels)
    
    vid = cv2.VideoCapture('data/test.avi')
    i = 0
    start = 0
    while True:
        read, frame = vid.read()
        if not read or i==4500+1:
            break

        name = 'frame'+str(i)+'.jpg'
        cv2.imwrite('data/train/'+name, frame)

        print('reading frame %i...' % i)
        for j in range(start,len(frame_labels)):
            frame_label = frame_labels[j]
            if(frame_label[1] != i):
                start = j
                break
            #do logic
            final_label = []
            final_label.append('../data/train/'+name)
            #write boundaries
            for x in range(8,12):
                final_label.append(str(int(frame_label[x])))
            final_label.append('person')
            print(final_label)
            final_labels.append(final_label)

        i += 1
    with open('data/train.csv', 'w') as csvfile:
        fw = csv.writer(csvfile)
        for label in final_labels:
            fw.writerow(label)

def read_csv(path):
    data = []
    with open(path, 'r') as f:
        r = csv.reader(f)
        for row in r:
            row = list(map(float, row))
            data.append(row)
    return data

create_data()
