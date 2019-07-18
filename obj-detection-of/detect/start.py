import numpy as np
import cv2
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
parser.add_option('--width', dest='width', help='Width of video.', default=960)
parser.add_option('--height', dest='height', help='Height of video.', default=540)

(options,args) = parser.parse_args()

w = int(options.width)
h = int(options.height)

d = Detection(w,h)
disp = Display(w,h)

writing = options.writing
write_frames = int(options.write_frames)
fourcc = cv2.VideoWriter_fourcc(*'H264')
wvid = cv2.VideoWriter('detection_of.mp4',fourcc,15.0,(w,h))

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
    out = d.detect(pimg, cimg)
    disp.paint(out)
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
