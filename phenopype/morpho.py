# -*- coding: utf-8 -*-
"""
Created: 2016/03/31
Last Update: 2018/10/02
Version 0.4.7
@author: Moritz Lürig
"""

#%% import

import os
import copy
import cv2
import numpy as np
import pandas as pd
import sys

from phenopype.utils import (blue, green, red, black, white)

#%% modules

class landmark_module:
    def __init__(self):

        # initialize # ----------------
        self.done = False 
        self.current = (0, 0) 
        self.landmarks = []
        self.idx = 0
        self.idx_list = []
        self.ref = False
        
    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)          
        if event == cv2.EVENT_LBUTTONDOWN and cv2.waitKey(1) & 0xff == 32: 
            print("Landmark reference with position (x=%d,y=%d) added" % (x, y))
            self.landmark_ref = (x, y)
            self.ref = True
        if event == cv2.EVENT_LBUTTONDOWN:
            self.landmarks.append((x, y))
            self.idx += 1
            self.idx_list.append(self.idx)
            print("Landmark #%d with position (x=%d,y=%d) added" % (self.idx, x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.landmarks) > 0:
                self.landmarks = self.landmarks[:-1]
                self.idx -= 1
                self.idx_list = self.idx_list[:-1]
                print("Landmark #%d with position (x=%d,y=%d) deleted" % (self.idx, x, y))
            else:
                print("No landmarks to delete")

    def draw(self, image, **kwargs):
        
        if isinstance(image, str):
            self.image = cv2.imread(image)
            self.filename = os.path.basename(image)
        elif isinstance(image, (list, tuple, np.ndarray)):
            self.image = image
            
        size = kwargs.get("size", int(((self.image.shape[0]+self.image.shape[1])/2)/150))
        col = kwargs.get("col", green)
        
        if not len(self.image.shape)==3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            
        if kwargs.get("zoom", False):
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            (rx,ry,w,h) = cv2.selectROI("phenopype", self.image, fromCenter=False)
            cv2.destroyWindow("phenopype")  
            if any([cv2.waitKey(50) & 0xff == 27, cv2.waitKey(50) & 0xff == 13]):
                cv2.destroyWindow("phenopype")  
            #self.points = [(x, y), (x, y+h), (x+w, y+h), (x+w, y)]
            temp_canvas1 = self.image[ry:ry+h,rx:rx+w]
            temp_canvas2 = temp_canvas1
        else:
            temp_canvas1 = copy.deepcopy(self.image)
            temp_canvas2 = temp_canvas1

        # =============================================================================
        # add points
        # =============================================================================
        print("\nAdd landmarks by left clicking, remove by right clicking, finish with enter.")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self.on_mouse)
        
        while(not self.done):
            
            if self.ref == True:
                cv2.circle(temp_canvas2, self.landmark_ref, size, red, -1)

            if self.idx > 0:
                for points, idx in zip(self.landmarks, self.idx_list):
                    cv2.circle(temp_canvas2, points, size, col, -1)
                    cv2.putText(temp_canvas2,  str(idx), points, cv2.FONT_HERSHEY_SIMPLEX, size/10, white,3,cv2.LINE_AA)

                    
            cv2.imshow("phenopype", temp_canvas2)
            
            if cv2.waitKey(50) & 0xff == 13:
                 self.done = True
                 cv2.destroyWindow("phenopype")
                 break
            elif cv2.waitKey(50) & 0xff == 27:
                cv2.destroyWindow("phenopype")
                break
                sys.exit("phenopype process stopped") 
                
            temp_canvas2 = copy.deepcopy(temp_canvas1)
            
            self.drawn = temp_canvas2

            if self.idx > 0:
                if self.ref == False:
                    self.landmark_ref = self.landmarks[0]
                    
                self.df = pd.DataFrame(data=self.landmarks, columns = ["x","y"], index=list(range(1,self.idx+1)))
                self.df["idx"] = self.idx_list
                self.df["filename"] = self.filename
                self.df["ref"] = str(self.landmark_ref)
                self.df = self.df[["filename", "idx", "x","y","ref"]]
     
   
class area_module:
    def __init__(self):

        # initialize # ----------------
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.idx = 0

    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.idx += 1
            print("Adding point #%d with position(%d,%d) to arena" % (self.idx, x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points = self.points[:-1]
                self.idx -= 1
                print("Removing point #%d with position(%d,%d) from arena" % (self.idx, x, y))
            else:
                print("No points to delete")

    def draw(self, image_path, mode,**kwargs):
        image = cv2.imread(image_path)
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print("\nMark the outline of your area by left clicking, delete with rightclick, finish with enter.")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self.on_mouse)
        temp_canvas = copy.deepcopy(image)
        
        if kwargs.get("zoom", False):
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            (rx,ry,w,h) = cv2.selectROI("phenopype", image, fromCenter=False)
            cv2.destroyWindow("phenopype")  
            if any([cv2.waitKey(50) & 0xff == 27, cv2.waitKey(50) & 0xff == 13]):
                cv2.destroyWindow("phenopype")  
            #self.points = [(x, y), (x, y+h), (x+w, y+h), (x+w, y)]
            temp_canvas1 = image[ry:ry+h,rx:rx+w]
            temp_canvas2 = temp_canvas1
        else:
            temp_canvas1 = copy.deepcopy(image)
            temp_canvas2 = temp_canvas1

        
        # =============================================================================
        # draw area
        # =============================================================================
    
        if mode == "polygon":
            while(not self.done):
                if (len(self.points) > 0):
                    cv2.polylines(temp_canvas1, np.array([self.points]), False, green, 3)
                    cv2.line(temp_canvas1, self.points[-1], self.current, blue, 3)
                cv2.imshow("phenopype", temp_canvas)
                temp_canvas1 = copy.deepcopy(temp_canvas2)
                
                if cv2.waitKey(50) & 0xff == 13:
                     self.done = True
                     cv2.destroyWindow("phenopype")
                     break
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
                     
        # MASK
        zeros = np.zeros(image.shape[0:2], np.uint8)
        zeros.fill(255)
        self.mask = cv2.fillPoly(zeros, np.array([self.points]), black)
        
        # DRAWN IMAGE
        boo = np.array(self.mask, dtype=bool)
        temp_canvas[boo,2] =255
        self.drawn = temp_canvas
        
        # ZOOM
        rx,ry,w,h = cv2.boundingRect(np.array(self.points, dtype=np.int32))
        self.image = image[ry:ry+h,rx:rx+w]

            
        
        if kwargs.get('show', False) == True:
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("phenopype", temp_canvas)
            cv2.waitKey(0)
            cv2.destroyWindow("phenopype")

        else:
            cv2.waitKey(1)
            cv2.destroyAllWindows()

#        cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
#        cv2.imshow('phenopype',  img)
