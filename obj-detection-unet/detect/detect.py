import cv2
import time
import numpy as np
import csv
from display import Display
from predict import Predictor

from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils

class Detection():
    def __init__(self, w, h, og_w, custom, obj_name, is_ofcalc=False):
        self.cheat_data = self.read_csv('../detector/data/data.csv')
        
        self.obj_name = obj_name
        self.c_start = 0
        self.prev_imgs = []

        self.resize_w = w
        self.resize_h = h
        self.multi = w/og_w
        
        if custom:
            self.start_idx = 1
            self.label_idx = 0
        else:
            self.start_idx = 8
            self.label_idx = 1

        self.is_ofcalc = is_ofcalc

        self.p = Predictor(self.resize_w, self.resize_h, 'models/model_detectorv2.h5')

        self.of_w = 376
        self.of_h = 240
        self.skipped_frames = 3

        self.rect_color = (10,125,10)
        self.font_size = 1
        self.font_thickness = 1

    def calc_of(self, pimg, cimg):
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
            flow = self.calc_of(self.prev_imgs[0], img)
        self.prev_imgs.append(img) 

        for i in range(self.c_start, len(self.cheat_data)):
            label = self.cheat_data[i]
            if label[self.label_idx] > nframe: #differnt for town data set
                self.c_start = i
                break

            #different for town data set
            #find cleaner way to instantiate this
            rect = [(int(label[self.start_idx]*self.multi) if label[self.start_idx] > 0 else 0, 
                    int(label[self.start_idx+1]*self.multi) if label[self.start_idx+1] > 0 else 0), 
                    (int(label[self.start_idx+2]*self.multi) if label[self.start_idx+2] > 0 else 0, 
                    int(label[self.start_idx+3]*self.multi) if label[self.start_idx+3] > 0 else 0)]
           
            moving, color = self.is_moving(flow, rect, type='color')

            cv2.rectangle(clone, rect[0], rect[1], color, 2) 
            self.prev_img = img
        return clone        

    def detect_nn(self, img):
        img = self.adjust_gamma(img, 0.4) #ONLY FOR THE BALLS
        mask_image = cv2.resize(self.p.detect_object(img), (self.resize_w, self.resize_h))
        contours, heir = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detection = img.copy()

        flow = []
        if len(self.prev_imgs) == self.skipped_frames:
            if self.is_ofcalc == True:
                flow = self.calc_of(self.prev_imgs[0], img)
            del self.prev_imgs[0]
        self.prev_imgs.append(img)

        #print(len(contours))
        for i in contours:
            if cv2.contourArea(i) < 1000:
                continue
            (x,y,w,h) = cv2.boundingRect(i)
            rect = [(x,y),(x+w,y+h)]
            moving, color = self.is_moving(flow, rect, type='color')

            cv2.rectangle(detection, rect[0], rect[1], color,2)

            moving_label = 'n/a'
            if moving==1:
                moving_label = 'moving'
            elif moving==0:
                moving_label = 'not moving'

            label = '{}: {}'.format(self.obj_name, moving_label)
            (retval,baseLine) = cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,self.font_size,self.font_thickness)
            textPos = (x,y)

            cv2.rectangle(detection, (textPos[0]-5, textPos[1]+baseLine-5), (textPos[0]+retval[0]+5, textPos[1]-retval[1]-5), (0,0,0), 2)
            cv2.rectangle(detection, (textPos[0]-5, textPos[1]+baseLine-5), (textPos[0]+retval[0]+5, textPos[1]-retval[1]-5), (255,255,255), -1)
            cv2.putText(detection, label, textPos, cv2.FONT_HERSHEY_DUPLEX, self.font_size, (0,0,0), self.font_thickness)
        #cv2.imshow('mask', mask_image)
        #cv2.waitKey(0)
        return detection

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

    def adjust_gamma(self, img, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(img, table)

