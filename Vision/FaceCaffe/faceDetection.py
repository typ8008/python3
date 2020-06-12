# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:47:10 2020

Face Detection.
It required following files to run:
    deploy.prototxt.txt
    res10_300x300_ssd_iter_140000.caffemodel
@author: Mariusz
"""


import numpy as np
import cv2 as cv
import sys

image = "../150.jpg"
proto = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
confidence = 0.4

img = cv.imread(image)


if img is None:
    print ("Error loading image")
    sys.exit(0)    
    
h, w = img.shape[:2]
# loading pretrained model
net = cv.dnn.readNetFromCaffe(proto, model)

# create blob
#blob = cv.dnn.blobFromImage(cv.resize(img, (300,300)), 1.0, (300,300),\
#                            (103.93, 116.77, 123.68))
blob = cv.dnn.blobFromImage(img, 1.0, (300,300),\
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
        cv.rectangle(img, (startX,startY), (endX, endY), (0,0,255),2)
        cv.putText(img, text, (startX + 2,y + 4), cv.FONT_HERSHEY_SIMPLEX, 0.6,\
                   (0,0,255),2)

cv.imshow("Image", img)
cv.waitKey()
cv.destroyAllWindows()