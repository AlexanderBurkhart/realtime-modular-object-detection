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

writing = False
write_frames = 1000
fourcc = cv2.VideoWriter_fourcc(*'H264')
wvid = cv2.VideoWriter('detection_of.mp4',fourcc,10.0,(w,h))

fvs = FileVideoStream('videos/test.avi').start() #faster than cv2 videowriter
print('Starting...')
time.sleep(1.0)

i = 0
fps = FPS().start()
while fvs.more():
    
    if i == write_frames and writing:
        break

    pimg = fvs.read()
    cimg = fvs.read()

    pimg = cv2.resize(pimg, (w,h))
    cimg = cv2.resize(cimg, (w,h))
    bgr = d.detect(pimg, cimg)
    disp.paint(bgr)
    fps.update()
    fps.stop()
    print('fps: %f' % fps.fps())
    if writing:
        wvid.write(out)
    i += 1

if writing:
    wvid.release()
    print('Saved video')
cv2.destroyAllWindows()
fvs.stop()
           
