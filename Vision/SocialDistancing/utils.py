# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:13:54 2020

@author: Mariusz
"""
import cv2 as cv
import numpy as np

def mouse_handler(event, x, y, flags, data):
    def draw():
        img = data['img'].copy()
        for point in data['points']:
            cv.circle(img, (point[0],point[1]), 3, (0,0,255), -1, 8)
            cv.putText(img, "({}, {})".format(point[0],point[1]), (point[0],point[1]+10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        
        # draw lines of points
        points = data['points']
        if (len(data['points'])) >= 2:
            cv.line(img,tuple(points[0]), tuple(points[1]), (0,255,0), 1, 8,0)
        
        if (len(data['points'])) >= 3:
            cv.line(img,tuple(points[1]), tuple(points[2]), (0,255,0), 1, 8,0)
        
        if (len(data['points'])) >= 4:
            cv.line(img,tuple(points[2]), tuple(points[3]), (0,255,0), 1, 8,0)
            cv.line(img,tuple(points[0]), tuple(points[3]), (0,255,0), 1, 8,0)
        
        cv.imshow("Image", img)  
    
    if event == cv.EVENT_LBUTTONDOWN:
        if len(data['points']) < 4:
            data['points'].append([x,y])
        draw()
    elif event == cv.EVENT_RBUTTONDOWN:
        if len(data['points']) != 0:
            data['points'].pop()
        draw()       
    
def get_four_points(image):
    data = {}
    data['img'] = image.copy()
    data['points'] = []
    
    cv.imshow("Image", image)
    cv.setMouseCallback("Image", mouse_handler, data)
    cv.waitKey()
    cv.destroyAllWindows()    
    points = np.vstack(data['points']).astype(float)
    
    return points
if __name__ == "__main__":
    frame = cv.imread("frame_0.jpg")
    print(get_four_points(frame))