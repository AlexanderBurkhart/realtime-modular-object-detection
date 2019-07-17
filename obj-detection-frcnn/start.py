import cv2
import time
import numpy as np
from display import Display
from detect import Detection

from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils


w = 960 #960
h = 540 #540

writing = False
write_frames = 100

d = Detection(w,h, zynq_support=False, using_nn=True)
disp = Display(w,h)

fourcc = cv2.VideoWriter_fourcc(*'H264')
wvid = cv2.VideoWriter('detection.mp4',fourcc,10.0,(w,h))

numframes = 0

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
    #out = d.cheat_detect(cimg, i)
    out = d.nn_detect(cimg)
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

