import cv2
import numpy as np
import time

from display import Display
from detect import Detection

from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils

fvs = FileVideoStream('test.avi').start() #faster than cv2 videowriter
print('Starting...')
time.sleep(1.0)

fps = FPS().start()
while fvs.more():
    cimg = fvs.read()
    cv2.imshow('img', cimg)
    cv2.waitKey(0)

    fps.update()
    fps.stop()
    print('fps: %f at frame %i' % (fps.fps(),i))
