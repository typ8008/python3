# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:23:29 2020

Camera Video Stream face detection.
It required following files to run:
    deploy.prototxt.txt
    res10_300x300_ssd_iter_140000.caffemodel
@author: Mariusz
"""


# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:47:10 2020

@author: Mariusz
"""

import numpy as np
import cv2 as cv
import sys

proto = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
confidence = 0.3


capture = cv.VideoCapture(0)

if not capture.isOpened():
    print ("error opening camera")
    sys.exit(0)
    
# loading pretrained model
net = cv.dnn.readNetFromCaffe(proto, model)

while True:
    ret, frame = capture.read()
    if ret == False:
        print("frame not captured,")
        sys.exit(0)

    h, w = frame.shape[:2]
    # create blob
    blob = cv.dnn.blobFromImage(frame, 1.0, (300,300),\
                            (103.93, 116.77, 123.68))

    # Pass blob to network and calculate network output    
    net.setInput(blob)
    detection = net.forward()

    # loop over detected faces and draw rectangle around

    for idx in range(detection.shape[2]):
        score = detection[0,0,idx,2]
        if  score > confidence:
            # coordinace of bounding box
            box = detection[0,0, idx, 3:7] * np.array([w,h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
        
            # draw box and probability
            text = "{:.2f}%".format(score*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.rectangle(frame, (startX,startY), (endX, endY), (0,0,255),2)
            cv.putText(frame, text, (startX,y), cv.FONT_HERSHEY_SIMPLEX, 0.45,\
                   (0,0,255),2)

    cv.imshow("Image", frame)

    if cv.waitKey(30) == 27:
        break
    
cv.destroyAllWindows()