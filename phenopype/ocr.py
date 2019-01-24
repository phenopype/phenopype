# -*- coding: utf-8 -*-
"""
Last Update: 2018/12/03
Version 0.4.7
@author: Moritz Lürig
"""

#%% import

import copy
import cv2
import math
import numpy as np
import sys

import pytesseract

from phenopype.utils import (red, green, blue, white)
from phenopype.utils import (blur)

#%% modules

class label_finder:
    def __init__(self):

        # setting up temporary variables
        self.done = False 
        self.current = (0, 0) 
        self.points = [] # List of points defining our polygon
        
    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point #%d to label outline" % (len(self.points)+1))
            self.points.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points = self.points[:-1]
                print("Removing point #%d from label outline" % (len(self.points)+1))

    def draw(self, image_path, mode, **kwargs): 
        
        # initialize # ----------------
        image = cv2.imread(image_path)       
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        

        # =============================================================================
        # dirdirdraw label outline
        # =============================================================================

        print("\nMark the outline of the label by left clicking, finish by right clicking:")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self.on_mouse)
        temp_canvas_1 = copy.deepcopy(image)
        
        if mode == "polygon":
            while(not self.done):
                if (len(self.points) > 0):
                    cv2.polylines(temp_canvas_1, np.array([self.points]), False, green, 2)
                    cv2.line(temp_canvas_1, self.points[-1], self.current, blue, 2)
                cv2.imshow("phenopype", temp_canvas_1)
                temp_canvas_1 = copy.deepcopy(image)
                if cv2.waitKey(50) & 0xff == 13:
                    self.done = True
                    cv2.destroyWindow("phenopype")
                elif cv2.waitKey(50) & 0xff == 27:
                    cv2.destroyWindow("phenopype")
                    break
                    sys.exit("phenopype process stopped")            
                
        elif mode == "box":
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            (x,y,w,h) = cv2.selectROI("phenopype", image, fromCenter=False)
            if any([cv2.waitKey(50) & 0xff == 27, cv2.waitKey(50) & 0xff == 13]):
                cv2.destroyWindow("phenopype")  
            self.points = [(x, y), (x, y+h), (x+w, y+h), (x+w, y)]


        # create colour mask to show scale outline
        colour_mask = np.zeros(image.shape, np.uint8)
        colour_mask[:,:,1] = 255 # all area green
        cv2.fillPoly(colour_mask, np.array([self.points]), red) # red = excluded area
        temp_canvas_1 = cv2.addWeighted(copy.deepcopy(image), .7, colour_mask, 0.3, 0) # combine
        
        
        # create template image for SIFT
        rx,ry,w,h = cv2.boundingRect(np.array(self.points, dtype=np.int32))
        self.image_original_template = image[ry:ry+h,rx:rx+w]
        
        # create template contour for SIFT
        cnt = np.array(self.points).reshape((-1,1,2)).astype(np.int32)
        self.box_original_template = cnt - cnt[0]

#        self.mask_original_template = np.zeros(image.shape[0:2], np.uint8)
#        cv2.fillPoly(self.mask_original_template, np.array([self.points]), white) 
#        self.mask_original_template = self.mask_original_template[ry:ry+h,rx:rx+w]

        if kwargs.get('show', True):
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("phenopype", self.image_original_template)    
            if any([cv2.waitKey(0) & 0xff == 27, cv2.waitKey(0) & 0xff == 13]):
                cv2.destroyWindow("phenopype")


    def find(self, image_path, **kwargs):
        
        # initialize ------------------
        self.image_target = cv2.imread(image_path)
        image_target = copy.deepcopy(self.image_target)
        image_original = copy.deepcopy(self.image_original_template)
        
        show = kwargs.get('show', False)
        min_matches = kwargs.get('min_matches', 10)
        ret = kwargs.get('ret', False)

        if (image_target.shape[0] + image_target.shape[1])/2 > 2000:
            factor = kwargs.get('resize', 0.5)
            image_target = cv2.resize(image_target, (0,0), fx=1*factor, fy=1*factor) 
            self.resized = True
        else:
            self.resized = False
        
        if not len(image_target.shape)==3:
            image_target = cv2.cvtColor(image_target, cv2.COLOR_GRAY2BGR)
                    
        # =============================================================================
        # AKAZE detector
        # =============================================================================
        
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(image_original,None)
        kp2, des2 = akaze.detectAndCompute(image_target,None)       
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        matches = matcher.knnMatch(des1, des2, 2)

        # keep only good matches
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        self.nkp = len(good)
        
        # find and transpose coordinates of matches
        if self.nkp >= min_matches:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            
            self.box_target = cv2.perspectiveTransform(self.box_original_template.astype(np.float32),M).astype(np.int32)
            
            image_target = cv2.polylines(image_target,[self.box_target],True,red,5, cv2.LINE_AA)
            
            if show:
                cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
                cv2.imshow("phenopype", image_target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if self.resized:
                self.box_target = self.box_target/factor                
            self.box_target = self.box_target.astype(np.int32)
            
            # MASK TARGET IMAGE
            zeros = np.zeros(self.image_target.shape[0:2], np.uint8)
            self.image_mask = cv2.fillPoly(zeros, [np.array(self.box_target, dtype=np.int32)], white)

            # SNIPPET FROM TARGET 
            (rx,ry,w,h) = cv2.boundingRect(self.box_target)
            self.label_image = self.image_target[max(0,ry): max(0,ry+h), max(0,rx): max(0,rx+w)]

            
            print("\n")
            print("--------------------------------------")
            print("Label found with %d keypoint matches" % self.nkp)
            print("--------------------------------------")
            print("\n")

            if ret:
                return self.label_image, self.image_mask      
        
        else:
            print("\n")
            print("----------------------------------------------")
            print("Label not found - only %d/%d keypoint matches" % (self.nkp, min_matches))
            print("----------------------------------------------")
            print("\n")
            
            if ret:
                return "no current label", "no label mask"


    def recognition(self, **kwargs):
        
        # initialize -----
        image = kwargs.get("image", self.label_image)
        blur_kernel = kwargs.get("blur", 15)
        offset = kwargs.get("offset", 25)
        if "config" in kwargs:
            config = kwargs.get("config")

        
        ret = kwargs.get('ret', False)
        show = kwargs.get("show", False)
        
        # blur + threshold 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = blur(gray, blur_kernel)
        ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # angle of non-white area
        coords = np.column_stack(np.where(thresh < 255))
        self.angle = cv2.minAreaRect(coords)[-1]
        if self.angle < -45:
        	self.angle = -(90 + self.angle)
        else:
            self.angle = -self.angle
        
        
        # expand label image to largest possible extensions 
        height, width = image.shape[:2]
        image_center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, self.angle, 1)
        
        radians = math.radians(self.angle)
        sin = math.sin(radians)
        cos = math.cos(radians)
        
        bound_w = int((height * abs(sin)) + (width * abs(cos)))
        bound_h = int((height * abs(cos)) + (width * abs(sin)))
        
        rotation_matrix[0, 2] += ((bound_w / 2) - image_center[0])
        rotation_matrix[1, 2] += ((bound_h / 2) - image_center[1])        
              
        # rotate!
        self.rotated_thresh = cv2.warpAffine(thresh, rotation_matrix, (bound_w, bound_h), borderValue = white)

        # blur + threshold rotated image
        coords = np.column_stack(np.where(self.rotated_thresh < 255))
        rect = cv2.minAreaRect(coords)
        
        ry = max(0, int((rect[0][0]-rect[1][0]/2)-offset))
        rx = max(0, int((rect[0][1]-rect[1][1]/2)-offset))
        h = int(rect[1][0]+2*offset)
        w = int(rect[1][1]+2*offset)            
        
        # crop rotated image
        self.rotated_label_image = self.rotated_thresh[ry:ry+h,rx:rx+w]
        
        if show:
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("phenopype", self.rotated_thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
        # ocr with tesseract
        if "config" in kwargs:
            self.label_string = pytesseract.image_to_string(self.rotated_label_image, config=config)
        else:
            self.label_string = pytesseract.image_to_string(self.rotated_label_image)

        if ret: 
            return print(self.label_string)