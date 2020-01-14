import cv2
import numpy as np
import time

from display import Display
from detect import Detection

from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--wr', '--writing', dest='writing', help='Is writing video or not.', default=False)
parser.add_option('--wf', '--write_frames', dest='write_frames', help='How many frames to write to video.', default=-1)
parser.add_option('--width', dest='width', help='Width of output video.', default=960)
parser.add_option('--height', dest='height', help='Height of output video.', default=540)
parser.add_option('--og_width', dest='og_width', help='Width of original video (needed if resizing output image or if image is not 1080p)', default=1920)
parser.add_option('--custom', dest='custom', help='Bool if detecting from custom data set or not.', default=False)
parser.add_option('--obj_name', dest='obj_name', help='Name of detected object.', default='object')
parser.add_option('--is_ofcalc', dest='is_ofcalc', help='Bool if calculating optical flow.', default=False)

(options,args) = parser.parse_args()

w = int(options.width)
h = int(options.height)
og_w = int(options.og_width)
custom = bool(options.custom)
obj_name = str(options.obj_name)
is_ofcalc = bool(options.is_ofcalc)

d = Detection(w,h,og_w,custom,obj_name,is_ofcalc)
disp = Display(w,h)

writing = options.writing
write_frames = int(options.write_frames)
fourcc = cv2.VideoWriter_fourcc(*'H264')
wvid = cv2.VideoWriter('detection_unet.mp4',fourcc,30.0,(w,h))

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
 
    #TODO: SEGMENTATION FAULT WITHOUT THESE TWO LINES
    cv2.imshow('out', out)
    cv2.waitKey(0)
    
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

