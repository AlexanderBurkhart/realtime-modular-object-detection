import cv2
import time
import numpy as np
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
parser.add_option('-n', '--nn', dest='using_nn', help='Using neural network.', default=True)
parser.add_option('-z', '--zynq', dest='zynq_support', help='Using the zynq.', default=False)

(options,args) = parser.parse_args()

w = int(options.width)
h = int(options.height)

d = Detection(w,h, zynq_support=options.zynq_support, using_nn=options.using_nn)
disp = Display(w,h)

writing = options.writing
write_frames = int(options.write_frames)
fourcc = cv2.VideoWriter_fourcc(*'H264')
wvid = cv2.VideoWriter('detection_frcnn.mp4',fourcc,30.0,(w,h))

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

