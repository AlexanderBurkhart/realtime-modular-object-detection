import numpy as np
import cv2
from keras.models import load_model
from keras import backend as K

SMOOTH = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

class Predictor():
    def __init__(self, w, h, modelpath):
        self.detector_model = load_model(modelpath, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
        self.detector_model._make_predict_function()
        
        self.resize_width = 384
        self.resize_height = 288

        self.resize_height_ratio = h/float(self.resize_height)
        self.resize_width_ratio = w/float(self.resize_width)
        self.middle_col = self.resize_width/2
        self.projection_threshold = 2
        self.projection_min = 200

    def extract_image(self, mask, img):
        pass 

    def detect_object(self, img):
        resize_img = cv2.cvtColor(cv2.resize(img, (self.resize_width, self.resize_height)), cv2.COLOR_BGR2GRAY)
        resize_img = resize_img[..., np.newaxis]
	
        img_mask = self.detector_model.predict(resize_img[None,:,:,:], batch_size=1)[0]
        img_mask = (img_mask[:,:,0]*255).astype(np.uint8)
        #print(img_mask)
        return img_mask
