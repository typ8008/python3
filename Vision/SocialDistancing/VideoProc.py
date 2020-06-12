# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:20:20 2020
Class for the social distancing violation detection
Camera is calibrated-ish for hte fixed camera for oxford high street dataset
for yolo algorithm people detection 3 files are required:
    coco.names - containing classified description
    yolov3.cfg - yolov3 configuration file
    yolov3.weights - pretrained network weights based on coco classifiers
    
@author: Mariusz
"""
import cv2 as cv
import numpy as np
import sys
from scipy.spatial import distance as dist
from utils import get_four_points

class VideoProc:
    
    def __init__(self, video):
        """
        constructor to set video  capture from the given path
        """
        self._capture = cv.VideoCapture(video)
                # pixel radius equivalent to 1 m
        self._RADIUS = 100
        # initialise HOG descriptor
        self._hog = cv.HOGDescriptor()
        # initialise People detection
        self._hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        
        # initialise yolo
        self._LABELS_PATH = "yolo\coco.names"
        self._CONFIG_PATH =  "yolo\yolov3.cfg"
        self._WEIGHTS_PATH = "yolo\yolov3.weights"
        self._CONFIDENCE = 0.5
        self._THRESHOLD = 0.3
        # load classification to list
        with open(self._LABELS_PATH) as file:
            self._LABELS = file.read().splitlines()
            
        # initialize random list of colours for each class for drawing purposes
        np.random.seed(42)
        self._COLORS = np.random.randint(0, 255, size = (len(self._LABELS),3), dtype=np.uint8)
    
        # load yolo object detector trained on COCO dataset
        self._net = cv.dnn.readNetFromDarknet(self._CONFIG_PATH, self._WEIGHTS_PATH)
        
        # determine output layer names
        self._ln = self._net.getLayerNames()
        self._ln = [self._ln[i[0] - 1 ] for i in self._net.getUnconnectedOutLayers()]
        
    def check_stream(self):
        """
        Check if the Video Stream can be accessed.
        Returns False if not and True if it's fine
        """
        if not self._capture.isOpened():
            print("Error Opening Video Strean")
            return False
        return True
    
    def get_stream(self):
        """
        Return video stream
        """
        return self._capture
    
    def get_frames(self,num_frames):
        """
        Input: takes number of frames from the video 
        Output: return frames as a list
        """
        frame_list = []
        # check if video stream available
        if self.check_stream():
            for idx in range(num_frames):
                ret, frame = self._capture.read()
                # check if no error
                if ret:
                    frame_list.append(frame)
        
        return frame_list
    
    def save_frames(self, frames, location = ""):
        """
        Input: Takes frames and location
        Outpu: write frames in the choosen location
        """
        for idx, frame in enumerate(frames):
            cv.imwrite(location + "frame_" + str(idx) + ".jpg", frame)
    
    def four_point_transform(self, image, pts):
        """
        Input: Image and points
        Output: Perspective transform
        """
        (tl, tr, br, bl) = pts

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = min(int(widthA), int(widthB))
        
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = min(int(heightA), int(heightB))       
                
        # Added offset to transformation points
        dst = np.array([
            [tl[0],bl[1] + 50],
            [maxWidth - 1 + tl[0], bl[1] +50],
            [maxWidth - 1 + tl[0], maxHeight-1 + bl[1] + 50],
            [tl[0], maxHeight - 1 + bl[1] + 50]
            ], dtype=np.float32)
        
        # Both function give the same result
        #M = cv.getPerspectiveTransform(rect,dst)        
        M, mask = cv.findHomography(pts, dst)
       
        #warped = cv.warpPerspective(image, M,(2000, 2100))

        return M

    def get_warped_image(self, image, M, size):
        """
        Input: image, homography matrix and size
        Output: warped imgage
        """
        # get warped image
        return cv.warpPerspective(image, M, (2000,2100))
        
    def bird_eye_view(self, image, M, warped, method = "HOG"):
        """
        Input: frame/image, Homography matrix and warped image
        Output: birth eye view image with original image 
        """
        
        # get point and boxes
        if method == "YOLO":
            boxes = self._yolo_detector(image)
        else:
            boxes = self._detect_people_HOG(image)
        
                
        points = []
        # get center points of the detected people
        for (x1, y1, x2, y2) in boxes:
            points.append(((x2 - x1)//2 + x1, (y2-y1)//2 + y1))

        # Create Black image, the same shape as warped
        black = np.zeros_like(warped)
    
        # transform center points based on Homography   
        center = np.float32(points).reshape(-1, 1, 2)
        trans = cv.perspectiveTransform(center, M)
        trans = trans.reshape(trans.shape[0],-1)
        trans = trans.astype(int)
        
        # detect violations
        violations = self._get_distance_violation(trans, self._RADIUS)
        
        # draw tranformed points and radius
        for idx, t in enumerate(trans[0:]):
            if idx in violations:
                color = (0,0,255)
            else:
                color = (255,255,0)
                
            cv.circle(black,(t[0],t[1]), 10, color, -1, 8)
            cv.circle(black,(t[0],t[1]), self._RADIUS, color, 10, 8)
            (x1, y1, x2, y2) = boxes[idx]
            cv.rectangle(image,(x1, y1), (x2, y2), color, 2)
            cv.circle(image,((x2 - x1)//2 + x1,(y2-y1)//2 + y1), 4, color, -1, 8)
            
        # Resize Original Image
        aspect = image.shape[0]/image.shape[1]
        image = cv.resize(image,(int(800/aspect), 800),\
                          interpolation = cv.INTER_AREA )
        
        # crop useful information
        black = black[:, 650:-75]
        
        # change resolution
        aspect = black.shape[0]/ black.shape[1]
        black = cv.resize(black, (int(640/aspect),640),\
                            interpolation = cv.INTER_AREA)
        
        # add border to bird eye image
        top = (image.shape[0] - black.shape[0]) //2
        bottom = top
        if bottom + top + black.shape[0] != image.shape[0]:
            bottom = image.shape[0] - black.shape[0]- top
        left = 50
        right = left
        addon = cv.copyMakeBorder(black, top, bottom, left, right, \
                                  cv.BORDER_CONSTANT, None, (250,218,133))
        
        # add description text 
        cv.putText(addon, "Bird's Eye View", (45, 50), cv.FONT_HERSHEY_PLAIN,\
                   3, (255,255,255), 4, 8)

        cv.putText(addon, "Violations: {}".format(len(violations)//2),\
                   (45, 770), cv.FONT_HERSHEY_PLAIN,\
                   2.5, (255,255,255), 4, 8)
        # combine image with bird eye view
        social = np.concatenate([image,addon], axis = 1) 
        
        return social
    
    def _detect_people_HOG(self, image):
        """
        Input: Image
        Output: coordinates of detected people using HOG Classifier
        """
        # grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
        # detect people in the frame and get boxes
        boxes, weights = self._hog.detectMultiScale(gray, winStride=(8,8))
    
        # convert list of boxes with coordinates
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        return boxes
        
    def _yolo_detector(self, image):
        """
        Input: image, paths to pre-trained algorithm labels, configuration 
        Output: coordinates of detected objects
        """
       
        # get image dimension
        H, W = image.shape[:2]
        
        # get blob
        blob = cv.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True,\
                                    crop = False)
    
        # set blob in network and execute nn calculation
        self._net.setInput(blob)
        layerOutputs = self._net.forward(self._ln)
        
        # initialise list of detected objects
        boxes = []
        confidences = []
        classIDs = []
        
        # loop over each layer outputs
        for output in layerOutputs:
            # loop over each of the detection
            for detection in output:
                # extract classID and confidence of detected objects
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                # filter out weak detections and only interested in people
                if confidence > self._CONFIDENCE and classID == 0:
                    # scale bounding box coordinates relative to size of the 
                    # image,Yolo returns centers (x, y) followed by the boxes
                    # width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    
                    # use center (x,y) coordinates to derive the top and left
                    # corners of the bounding box
                    x = int(centerX - (width /2))
                    y = int(centerY - (height/2))
                    
                    # update lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression, Removes overlapping boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, self._CONFIDENCE, self._THRESHOLD)
        filtered_boxes = []
        # get  filtered boxes
        if len(idxs) > 0:
            # loop over indexes which are kept
            for i in idxs.flatten():
                # extract boxes
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                
                filtered_boxes.append((x, y, x + w, y + h))
        
        filtered_boxes = np.asarray(filtered_boxes)
        return filtered_boxes
    
    def _get_distance_violation(self, location, distance):
        """
        Inpput: locations and distance between people
        Output: return indexes of the violations
        """
        # get eucledean distance between all the center points
        eucl_dist = dist.cdist(location, location, metric="euclidean")
        violation = set()
        # Loop over upper matrix distance
        for i in range(eucl_dist.shape[0]):
            # loop over lower matrix distance apart from itself
            for j in range(i+1, eucl_dist.shape[1]):
                # check if the distance is less then predefined
                if eucl_dist[i,j] < distance * 2 :
                    # add indexes to the set
                    violation.add(i)
                    violation.add(j)        

        return violation
    
    def run(self, frame_num, save, method = "HOG"):
        """
        Input: Method for people detection, number of frames to record, if -1
        then all of them will be recorded, 
        save flag, if True save video, otherwise play
        saved processed video
        """
        if save:
            fps = int(self._capture.get(cv.CAP_PROP_FPS))
            out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'),\
                             fps, (1910,800))
        
        print (frame_num)
        if frame_num > self._capture.get(cv.CAP_PROP_FRAME_COUNT) or frame_num < 0:
            frame_num = int(self._capture.get(cv.CAP_PROP_FRAME_COUNT))
        
        frame = cv.imread("frame_0.jpg")
        M = self.four_point_transform(frame,\
                np.array([(1116,219),\
                          (1536,263),\
                          (930, 883),\
                          (292,735)], dtype=np.float32))
        warped = self.get_warped_image(frame, M, (2000,2100))
        
        for _ in range(frame_num):
            ret, frame = self._capture.read()
        
            if ret == True:
                social =  self.bird_eye_view(frame, M, warped, method)
                # if save flag is on then save otherwise display
                if save:
                    out.write(social)
                else:
                    cv.imshow("social", social)   
                if cv.waitKey(1) & 0xff == 27:
                    break
            else:
                break
            
        self._capture.release()
        if save:
            out.release()     

if __name__ == "__main__":    
    video = VideoProc("TownCentreXVID.avi")
    
    frame = cv.imread("frame_0.jpg")
    get_four_points(frame)
    M = video.four_point_transform(frame,\
                np.array([(1116,219),\
                          (1536,263),\
                          (930, 883),\
                          (292,735)], dtype=np.float32))
    warped = video.get_warped_image(frame, M, (2000,2100))
    capture = video.get_stream()
    if video.check_stream() == False:
        print("video stream unavailable")
        sys.exit(-1)
     
    video.run(100, False, "HOG")    

    cv.destroyAllWindows()