import numpy as np
import cv2
import time
from display import Display
from detect import Detection

from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils

w = 960
h = 540

d = Detection()
disp = Display(w,h)

numframes = 0
start = time.time()

fvs = FileVideoStream('videos/test.avi').start() #faster than cv2 videowriter
print('Starting...')
time.sleep(1.0)

fps = FPS().start()
while fvs.more():
    pimg = fvs.read()
    cimg = fvs.read()

    pimg = cv2.resize(pimg, (w,h))
    cimg = cv2.resize(cimg, (w,h))
    bgr = d.detect(pimg, cimg)
    disp.paint(bgr)
    fps.update()
    fps.stop()
    print('fps: %f' % fps.fps())
cv2.destroyAllWindows()
fvs.stop()
           
