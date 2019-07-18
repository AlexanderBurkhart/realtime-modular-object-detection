import cv2
import time
import numpy as np
from display import Display

from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils

from keras.models import load_model

class Detection():
    def __init__(self, w, h):
        self.w = w
        self.h = h

        #self.model = load_model('../train/saved_models/modelv1.h5')
        self.nn_w = 256
        self.nn_h = 128

        self.of_w = w
        self.of_h = h

    def calc_of(self, pimg, cimg):

        pimg = cv2.resize(pimg, (self.of_w, self.of_h))
        cimg = cv2.resize(cimg, (self.of_w, self.of_h))

        hsv = np.zeros_like(pimg)

        pimg_gray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
        cimg_gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    
        flow = cv2.calcOpticalFlowFarneback(pimg_gray, cimg_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        ret,bgr = cv2.threshold(hsv[...,2],10,255,cv2.THRESH_BINARY)
        return bgr

    def detect(self, pimg, cimg):
        bgr = self.calc_of(pimg, cimg)
        bgr = cv2.resize(bgr, (self.w, self.h))
        _, contours, heir = cv2.findContours(bgr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detection = cimg.copy()
        for i in contours:
            if cv2.contourArea(i) < 500:
                continue
            (x,y,w,h) = cv2.boundingRect(i)
            cv2.rectangle(detection, (x,y), (x+w,y+h), (0,0,255),2) 
        return detection
  
    def detect_and_narrow(self, pimg, cimg):
        bgr = self.calc_of(pimg, cimg)
        _, contours, heir = cv2.findContours(bgr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detection = cimg.copy()
        crop_detects = []
        for i in contours:
            if cv2.contourArea(i) < 500:
                continue
            (x,y,w,h) = cv2.boundingRect(i)
            crop = cimg[y:y+h, x:x+w]
            crop = cv2.resize(crop, (self.nn_h, self.nn_w))
            crop_array = np.asarray(crop)
            pred = self.model.predict(crop_array[None,:,:,:], batch_size=32)
            #print('is pedestrian:{}'.format(pred))
            if pred[0][0] <= pred[0][1]:
                cv2.rectangle(detection, (x,y), (x+w,y+h), (0,0,255),2)
        return detection 
