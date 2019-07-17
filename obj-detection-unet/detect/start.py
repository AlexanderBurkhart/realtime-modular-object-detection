import cv2
import numpy as np
import time

from display import Display
from detect import Detection

from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils

w = 1024 #960
h = 576 #540

d = Detection(w,h)
disp = Display(w,h)

writing = False
write_frames = 1000
fourcc = cv2.VideoWriter_fourcc(*'H264')
wvid = cv2.VideoWriter('detection_unet.mp4',fourcc,10.0,(w,h))

fvs = FileVideoStream('videos/test.avi').start() #faster than cv2 videowriter
print('Starting...')
time.sleep(1.0)

i = 0
fps = FPS().start()
while fvs.more():

    if i == write_frames and writing:
        break

    cimg = fvs.read()

    cimg = cv2.resize(cimg, (w,h))
    out = d.detect_nn(cimg)
    #out = d.cheat_detect(cimg, i)
    disp.paint(out)
    fps.update()
    fps.stop()
    print('fps: %f at frame %i' % (fps.fps(),i))
    if writing:
        wvid.write(out)
    i += 1

if writing:
    wvid.release()
    print('Saved video.')

cv2.destroyAllWindows()
fvs.stop()

