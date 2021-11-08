#!/usr/bin/python3


import motorcortex
import time
import start
import cv2
from cv_bridge import CvBridge
import os
from math import cos, sin, sqrt
import numpy as np

class tc3():
    def __init__(self):
        self.test = start.Solution()
        # Creating empty object for parameter tree
        parameter_tree = motorcortex.ParameterTree()
        # Loading protobuf types and hashes
        motorcortex_types = motorcortex.MessageTypes()
        # Open request connection
        self.ip = "192.168.42.1"
        self.frame = "tracking_cam3"
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.req, self.sub = motorcortex.connect("ws://"+self.ip+":5558:5557", motorcortex_types, parameter_tree,
                                    certificate=dir_path+"/motorcortex.crt", timeout_ms=1000,
                                    login="root", password="vectioneer")

        self.bridge = CvBridge()
        
        self.subscription6 = self.sub.subscribe(["root/Comm_task/utilization_max","root/Processing/image"], "camera", 1)
        self.subscription6.get()
        self.subscription6.notify(self.onImage)
        #self.out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 1,(640,480))
        self.count = 1
      
    def onImage(self,val):
        image = cv2.imdecode(np.frombuffer(val[1].value, np.uint8), -1)
        image_original = image
        #print(type(image))
        path = 'data_base'

        #cv2.imwrite(os.path.join(path,'color_black_2' +str(self.count) +'.jpg'), image)
        #self.out.write(image)
        #print(image.shape)
        image = self.test.binareid(image)
        image = self.test.transformFunc(image)
        cv2.imshow('Frame', image)
        cv2.imshow('Frame_original', image_original)
        cv2.waitKey(1)
            
            

        #cv2.imshow('Frame', image)
if __name__ == '__main__':
    tc3_ex = tc3()
    while True:
        try:
           t = 0
        except Exception as e:
            print(e)
            tc3_ex.req.close()
            tc3_ex.sub.close()