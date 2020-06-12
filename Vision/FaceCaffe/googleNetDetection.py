# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:23:29 2020

@author: Mariusz
"""


# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:47:10 2020

Script to detect objects using Google Net classifier
    it required following files:
       bvlc_googlenet.prototxt
       bvlc_googlenet.caffemodel
       classification_classes_ILSVRC2012.txt
@author: Mariusz
"""

import numpy as np
import cv2 as cv
import sys

proto = "bvlc_googlenet.prototxt"
model = "bvlc_googlenet.caffemodel"
classification = "classification_classes_ILSVRC2012.txt"
confidence = 0.3
mean = (104, 117, 123)
capture = cv.VideoCapture(0)
size = (224,224)
swapRB = False
classes = []
 
if not capture.isOpened():
    print ("error opening camera")
    sys.exit(0)
    
# loading pretrained model
net = cv.dnn.readNet(model, proto)

# load classification to list
with open(classification) as file:
    classes = file.read().splitlines()

while True:
    ret, frame = capture.read()
    if ret == False:
        print("frame not captured,")
        sys.exit(0)

    h, w = frame.shape[:2]
    # create blob
    blob = cv.dnn.blobFromImage(frame, 1.0, size, mean, swapRB, False)

    # Pass blob to network and calculate network output    
    net.setInput(blob)
    detection = net.forward()

    # calculate valcues and confidence
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(detection)
    
    # add text with description of the item in the video and percentage in confidence
    cv.putText(frame, classes[maxLoc[0]], (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7,\
                   (0,0,255),2)
        
    cv.putText(frame, "confidence: {:0.2f} %".format(maxVal * 100), (10,60),\
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    
    cv.imshow("Image", frame)
    if cv.waitKey(30) == 27:
        break
    
cv.destroyAllWindows()