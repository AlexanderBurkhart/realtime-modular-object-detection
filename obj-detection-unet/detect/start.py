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

(options,args) = parser.parse_args()

w = int(options.width)
h = int(options.height)
og_w = int(options.og_width)
custom = bool(options.custom)

d = Detection(w,h,og_w,custom)
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
   
    #start buffer (rough fix for race condition between opencv and imutils)
    if i==0:
        print('Press any key to start...')
        cv2.imshow('Buffer', np.zeros((100,100,3), np.uint8))
        cv2.waitKey(0)
        time.sleep(0.5)
        cv2.destroyAllWindows()

    out = d.detect_nn(cimg)
    #out = d.cheat_detect(cimg, i)
    disp.paint(out)
    fps.update()
    fps.stop()
    print('fps: %f at frame %i' % (fps.fps(),i))
    if writing:
        wvid.write(out)

    cv2.imshow('out', out)
    cv2.waitKey(0)

    i += 1

if writing:
    wvid.release()
    print('Saved video.')

cv2.destroyAllWindows()
fvs.stop()

