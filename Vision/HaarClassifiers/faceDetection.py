# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:23:34 2020
Haar classifier for face detection
More classifiers can be found here
https://github.com/opencv/opencv/tree/master/data/haarcascades
@author: Mariusz
"""

from __future__ import print_function
import cv2 as cv
import time
import numpy as np
import pkg_resources.py2_warn

def cascade_classifier():
    """
    object detection using pretrained cascade classifier
    """
    # get names of the files with pretrained models
    face_cascade_file = 'haarcascade_frontalface_alt2.xml'
    eyes_cascade_file = 'haarcascade_eye.xml'
    
    # define cascade classifiers
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    
    # Load pretranined models
    if not face_cascade.load(cv.samples.findFile(face_cascade_file)):
        print ("Error loading face cascade")
        return -1
    
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_file)):
        print ("Error loading eyes cascade")
        return -1
    
    # Open Camera stream
    capture = cv.VideoCapture(0)
    if not capture.isOpened:
        print ("Error opening camera")
        return -1
    
    
    while True:
        # capture frame 
        ret, frame = capture.read()
        if not ret:
            print ("Error capturing frame")
            break
        
        # convert to grayscae
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # get Histogram
        gray = cv.equalizeHist(gray)
        
        # detect face
        faces = face_cascade.detectMultiScale(gray)
        
        for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360,\
                               (255,0,255), 4)
        
            # replace face by a blur
            #face = frame[y: y + h, x: x + w
            #blur_face = cv.GaussianBlur(face,(25,25),0)
            #frame[y: y + h, x: x + w,:] = blur_face
            #cv.imshow("Color Face", blur_face)
            
            faceROI = gray[y: y + h, x: x + w]
            #cv.imshow("face", faceROI)
            
            # detect eyes in face
            #eyes = eyes_cascade.detectMultiScale(faceROI)
            #for (x2, y2, w2, h2) in eyes:
            #    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            #    radius = int(round((w2 + h2)/4))
            #    frame = cv.circle(frame, eye_center, radius, (255,0,0), 4)
        
         
        cv.imshow("Capture", frame)
        
        if cv.waitKey(20) == 27:
            break
        
    cv.destroyAllWindows()

if __name__ == "__main__":
    cascade_classifier()