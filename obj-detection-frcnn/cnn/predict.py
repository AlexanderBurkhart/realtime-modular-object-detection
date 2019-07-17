from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn

class Prediction():
    def __init__(self, config_path, model_path): 
        #init config param
        config_output_filename = config_path
        with open(config_output_filename, 'rb') as f_in:
            self.C = pickle.load(f_in)

        # turn off any data augmentation at test time
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_90 = False
        
        #init class maps
        #e.g. {0: "class", 1: "another class"}
        self.class_mapping = self.C.class_mapping
        if 'bg' not in self.class_mapping:
            self.class_mapping['bg'] = len(self.class_mapping)
        self.class_mapping = {v: k for k, v in self.class_mapping.items()}

        #init viz
        self.class_to_color = {self.class_mapping[v]: np.random.randint(0, 255, 3) for v in self.class_mapping}
        self.bbox_threshold = 0.85

        #init nn params
        self.C.num_rois = 16
        num_features = 1024
        
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)
        
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)
        
        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)
        
        # define the RPN, built on the base layers
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)
        
        classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.class_mapping), trainable=True)
        
        #init models
        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model([feature_map_input, roi_input], classifier)
        
        self.model_classifier = Model([feature_map_input, roi_input], classifier)
        
        #load weights
        print('Loading weights from {}'.format(model_path))
        self.model_rpn.load_weights(model_path, by_name=True)
        self.model_classifier.load_weights(model_path, by_name=True)
        
        #compile
        self.model_rpn.compile(optimizer='adam', loss='mse') #sgd
        self.model_classifier.compile(optimizer='adam', loss='mse')
    
    def format_img_size(self, img):
        """ formats the image size based on config """
        img_min_side = float(self.C.im_size)
        (height, width ,_) = img.shape        
        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio
        
    def format_img_channels(self, img):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self.C.img_channel_mean[0]
        img[:, :, 1] -= self.C.img_channel_mean[1]
        img[:, :, 2] -= self.C.img_channel_mean[2]
        img /= self.C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
        
    def format_img(self, img):
        """ formats an image for model prediction based on config """
        img, ratio = self.format_img_size(img)
        img = self.format_img_channels(img)
        return img, ratio
        
    # Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(self, ratio, x1, y1, x2, y2):
        
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))
        
        return (real_x1, real_y1, real_x2 ,real_y2)

    #TODO: this guy made this rly slow....fix it
    def predict(self, img):

        st = time.time()

        X, ratio = self.format_img(img)
    
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))
    
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = self.model_rpn.predict(X)
    
    
        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)
    
        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
    
        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
    
        for jk in range(R.shape[0]//self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
    
            if jk == R.shape[0]//self.C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
    
            [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])
    
            for ii in range(P_cls.shape[1]):
    
                if np.max(P_cls[0, ii, :]) < self.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue
    
                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]
    
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
    
                (x, y, w, h) = ROIs[0, ii, :]
    
                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
    
        all_dets = []
    
        for key in bboxes:
            bbox = np.array(bboxes[key])
    
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            return new_boxes, ratio
            
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
    
                (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
   
                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (10, 250, 10),2)
    
                #textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                #all_dets.append((key,100*new_probs[jk]))
    
                #(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                #textOrg = (real_x1, real_y1-0)
    
                #cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                #cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                #cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    
        #print('Elapsed time = {}'.format(time.time() - st))
        #print(all_dets)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        return img
