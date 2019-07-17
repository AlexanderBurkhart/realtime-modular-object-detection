import sys
sys.path.append('cnn')

import cv2
import time
import numpy as np
import csv
from display import Display
from server import Zynq_OF

class Detection():
    def __init__(self, w, h, zynq_support=False, using_nn=True):
        self.obj_name = 'person'
        
        self.p = None
        self.using_nn = using_nn
        if self.using_nn:
            from cnn.predict import Prediction
            self.p = Prediction('cnn/config.pickle', 'cnn/model_frcnn.hdf5')
        
        self.zynq = None
        self.zynq_support = zynq_support
        if self.zynq_support:
            self.zynq = Zynq_OF()
       
        self.cheat_data = self.read_csv('cnn/data/data.csv')
        self.c_start = 0
        self.prev_imgs = []

        self.resize_w = w
        self.resize_h = h
        self.multi = w/1920

        self.of_w = 376
        self.of_h = 240
        self.skipped_frames = 3

        self.rect_color = (10,125,10)
        self.font_size = 1
        self.font_thickness = 1

    def calc_of(self, pimg, cimg):
        if len(pimg)==0:
            return []
        p_resize = cv2.resize(pimg, (self.of_w, self.of_h))
        c_resize = cv2.resize(cimg, (self.of_w, self.of_h))
        pimg_gray = cv2.cvtColor(p_resize, cv2.COLOR_BGR2GRAY)
        cimg_gray = cv2.cvtColor(c_resize, cv2.COLOR_BGR2GRAY)
    

        hsv = np.zeros_like(p_resize)
    
        flow = cv2.calcOpticalFlowFarneback(pimg_gray, cimg_gray, None, 0.75, 3, 25, 3, 7, 1.2, 0)
    
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        ret,bgr = cv2.threshold(hsv[...,2],10,255,cv2.THRESH_BINARY)
        bgr = cv2.resize(bgr, (self.resize_w, self.resize_h))
        return bgr

    def cheat_detect(self, img, nframe):
        clone = img.copy()
        flow = []
        if len(self.prev_imgs) == self.skipped_frames:
            if self.zynq_support:
                flow = self.zynq.grab_OF()
            else:
                flow = self.calc_of(self.prev_imgs[0], img)
            del self.prev_imgs[0]
        self.prev_imgs.append(img)

        for i in range(self.c_start, len(self.cheat_data)):
            label = self.cheat_data[i]
            if label[1] > nframe:
                self.c_start = i
                break

            rect = [(int(label[8]*self.multi) if label[8] > 0 else 0, 
                    int(label[9]*self.multi) if label[9] > 0 else 0), 
                    (int(label[10]*self.multi) if label[10] > 0 else 0, 
                    int(label[11]*self.multi) if label[11] > 0 else 0)]
           
            moving, moving_color = self.is_moving(flow, rect, type='color')

            cv2.rectangle(clone, rect[0], rect[1], moving_color, 2) 
        return clone        

    def nn_detect(self, img):
        if not self.using_nn:
            raise Exception('This Detection object is not supported for neural networks. Set using_nn to True to use this function.')
        boxes,ratio = self.p.predict(img)
        flow = []
        if len(self.prev_imgs) == self.skipped_frames:
            if self.zynq_support:
                flow = self.zynq.grab_OF()
            else:
                flow = self.calc_of(self.prev_imgs[0], img)
            del self.prev_imgs[0]
        self.prev_imgs.append(img)

        for i in range(boxes.shape[0]):
            (x1,y1,x2,y2) = boxes[i,:]
            (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
            rect = [(real_x1 if real_x1 > 0 else 0, real_y1 if real_y1 > 0 else 0),
                    (real_x2 if real_x2 > 0 else 0 ,real_y2 if real_y2 > 0 else 0)]

            moving, color = self.is_moving(flow, rect, type='color')
            cv2.rectangle(img,rect[0], rect[1], color, 2)

            moving_label = 'n/a'
            if moving==1:
                moving_label = 'moving'
            elif moving==0:
                moving_label = 'not moving'

            label = '{}: {}'.format(self.obj_name, moving_label)
            (retval,baseLine) = cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,self.font_size,self.font_thickness)
            textPos = (real_x1,real_y1)
            
            cv2.rectangle(img, (textPos[0]-5, textPos[1]+baseLine-5), (textPos[0]+retval[0]+5, textPos[1]-retval[1]-5), (0,0,0), 2)
            cv2.rectangle(img, (textPos[0]-5, textPos[1]+baseLine-5), (textPos[0]+retval[0]+5, textPos[1]-retval[1]-5), (255,255,255), -1)
            cv2.putText(img, label, textPos, cv2.FONT_HERSHEY_DUPLEX, self.font_size, (0,0,0), self.font_thickness)

        return img

    # Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(self, ratio, x1, y1, x2, y2):

        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2 ,real_y2)

    #HAS ISSUES WHEN PERSON IS BEHIND AN OBSTACLE
    #fix: be able to have a mask around the pedestrain and not a bounding box
    def is_moving(self, flow, rect,  type='none'):
        if len(flow) == 0:
            #can't tell
            if type=='color':
                return -1, (self.rect_color[0]*2, self.rect_color[1], self.rect_color[2])
            return -1

        flow_crop = flow[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        white_count = cv2.countNonZero(flow_crop)

        flow_per = white_count/(flow_crop.shape[0]*flow_crop.shape[1])
      
        if flow_per > 0.2:
            #moving
            if type=='color':
                return 1, (self.rect_color[0], self.rect_color[1]*2, self.rect_color[2])
            return 1
        #not moving
        if type =='color':
            return 0, (self.rect_color[0] , self.rect_color[1], self.rect_color[2]*2)
        return 0

    def read_csv(self, path):
        data = []
        with open(path, 'r') as f:
                r = csv.reader(f)
                for row in r:
                    row = list(map(float, row))
                    data.append(row)
        return data
