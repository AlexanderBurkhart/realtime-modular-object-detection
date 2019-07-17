import socket
import cv2
import pickle
import struct ## new
import numpy as np

class Zynq_OF():

    def __init__(self):
 
        HOST=''
        PORT=8485
        
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        print('Socket created')
        
        s.bind((HOST,PORT))
        print('Socket bind complete')
        s.listen(10)
        print('Socket now listening')
        
        self.conn,addr=s.accept()
        
        self.data = b""
        self.payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(self.payload_size))


    def draw_hsv(self, frame):
        
        c = np.dsplit(frame,3)
         
        fx = c[0][:,:,0]
        fy = c[1][:,:,0]
    
        h, w = frame.shape[:2]
       
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,2] = np.minimum(v*4, 255) # keep this channel for grayscale
        ret,bgr = cv2.threshold(hsv[...,2],0,255,cv2.THRESH_BINARY)
                                
        return bgr
    
    def findMovement(self, bgr, outframe):
        kernel = np.ones((5,5), np.uint8)
        closing = cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, kernel)
        _, contours, heir = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        for i in contours:
            if cv2.contourArea(i) < 10:
                continue
            (x,y,w,h) = cv2.boundingRect(i)
            cv2.rectangle(outframe,(x,y), (x+w,y+h),(0,0,255),2)
    
    
    def grab_OF(self):
        while len(self.data) < self.payload_size:
            print("Recv: {}".format(len(self.data)))
            self.data += self.conn.recv(4096)
    
        #print("Done Recv: {}".format(len(self.data)))
        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        #print("msg_size: {}".format(msg_size))
        while len(self.data) < msg_size:
            self.data += self.conn.recv(4096)
        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]
    
        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
       
        frame1 = self.draw_hsv(frame) 
    
        #cv2.imshow('ImageWindow',frame1)
        #cv2.waitKey(0)
        return frame1
