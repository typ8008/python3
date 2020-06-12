# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:30:59 2020

@author: Mariusz
"""


import cv2 as cv
import sys

net = cv.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "frozen_inference_graph.pbtxt")

img = cv.imread("../150.jpg")
if img is None:
    print("image not loaded")
    exit(0)
    
rows, cols, channels = img.shape

# convert image to blob
input = cv.dnn.blobFromImage(img, size = (300,300), swapRB=True, crop = False)
net.setInput(input)

# run forward pass to compute network
output = net.forward()

# loop on the output
for detection in output[0,0]:
    score = float(detection[2])
    print ("Score", score)
    if score > 0.4:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)),(0,0,255))
        
cv.imshow("Image", img)
cv.waitKey()
cv.destroyAllWindows()