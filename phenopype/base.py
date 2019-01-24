# -*- coding: utf-8 -*-
"""
Last Update: 2019/01/14
Version 0.4.9
@author: Moritz Lürig
"""
#%% import 

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import copy
import math
import sys

import cv2
import datetime
from collections import Counter

from phenopype.utils import (red, green, blue, black, white)
from phenopype.utils import (blur, exif_date, gray_scale)

#%% classes
class project:      
    def project_maker(self, project_name, image_dir, **kwargs):
        """ 
        create project object where the results of the image analysis are saved
        Parameters
        ----------
        project_name: str
            name of your project
        image_dir: str 
            path to directory with images
            
        Parameters (optional)
        ---------------------
        exclude: list with str
            single or multiple string patterns to exclude certain files
        mode: str ("tree", "dir")
            tree mode loops through all subdirectories of the tree, dir only takes valid files from upper directory
        resize: int (>0 - 1)
            global resizing of images to increase performance (1=original size)      
        save_dir: str
            path to directory where processed images and measurements are saved
      
        """
        # initialize ----
        self.name = project_name
        self.in_dir = image_dir
        self.save_dir = kwargs.get("save_dir", os.path.normpath(self.in_dir) + "_out")   
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.mode = kwargs.get("mode","tree")         
        self.resize = kwargs.get("resize",1)
        
        self.filetypes = kwargs.get("filetypes", [])
        self.exclude = kwargs.get("exclude", [])
        self.include = kwargs.get("include", [])


        # MAKE FILELISTS
        filepaths1 = []
        filenames1 = []
        if self.mode == "tree":
            for root, dirs, files in os.walk(self.in_dir):
                for i in os.listdir(root):
                    path = os.path.join(root,i)
                    if os.path.isfile(path):      
                        filepaths1.append(path)
                        filenames1.append(i)
                        
        elif self.mode == "dir":
            for i in os.listdir(self.in_dir):
                path = os.path.join(self.in_dir,i)
                if os.path.isfile(path):      
                    filepaths1.append(path)
                    filenames1.append(i)
                     
        # INCLUDE        
        filepaths2 = []
        filenames2 = []
        if self.include:
            for name, path in zip(filenames1, filepaths1):   
                for inc in self.include:
                    if inc in name:
                        filepaths2.append(path)
                        filenames2.append(name)
        else:
            filepaths2 = filepaths1
            filenames2 = filenames1

        # EXCLUDE  
        filepaths3 = []
        filenames3 = []
        if self.exclude:
            for name, path in zip(filenames2, filepaths2):   
                for exc in self.exclude:
                    if exc not in name:
                        filepaths3.append(path)
                        filenames3.append(name)
        else:
            filepaths3 = filepaths2
            filenames3 = filenames2

        # FILETYPE        
        filepaths4 = []
        filenames4 = []
        if self.filetypes:
            for name, path in zip(filenames3, filepaths3):   
                for ext in self.filetypes:
                    if name.endswith(ext):
                        filepaths4.append(path)
                        filenames4.append(name)
        else:
            filepaths4 = filepaths3
            filenames4 = filenames3

        self.filenames = filenames4
        self.filepaths = filepaths4
                    
        # =============================================================================
        # BUILD PROJECT DF
        # =============================================================================
        self.df = pd.DataFrame(data=list(zip(self.filepaths, self.filenames)), columns = ["filepath", "filename"])
        self.df.index = self.filenames
        self.df.insert(0, "project", self.name)
        self.df.drop_duplicates(subset="filename", inplace=True)
        self.filepaths = self.df['filepath'].tolist()
        self.filenames = self.df['filename'].tolist()
        self.df.drop(columns='filepath', inplace=True)
        
    
#    def update_list(self, **kwargs):
#        
#        # =============================================================================
#        # MAKE FILELIST AND PATHLIST
#        # =============================================================================
#        filepaths1 = []
#        filenames1 = []
#        
#        if self.mode == "tree":
#            for root, dirs, files in os.walk(self.in_dir):
#                for i in os.listdir(root):
#                    path = os.path.join(root,i)
#                    if os.path.isfile(path):      
#                        filepaths1.append(path)
#                        filenames1.append(i)
#                        
#        elif self.mode == "dir":
#            for i in os.listdir(self.in_dir):
#                path = os.path.join(self.in_dir,i)
#                if os.path.isfile(path):      
#                    filepaths1.append(path)
#                    filenames1.append(i)
#                      
#        filepaths2 = []
#        filenames2 = []
#        for name, path in zip(filenames1, filepaths1):   
#            for inc in self.include:
#                if inc in name:
#                    filepaths2.append(path)
#                    filenames2.append(name)
#
#        filepaths3 = []
#        filenames3 = []
#        for name, path in zip(filenames2, filepaths2):   
#            for exc in self.exclude:
#                if exc not in name:
#                    filepaths3.append(path)
#                    filenames3.append(name)
#
#        filepaths4 = []
#        filenames4 = []
#        for name, path in zip(filenames3, filepaths3):   
#            if name.endswith in self.file_type:
#                filepaths4.append(path)
#                filenames4.append(name)
#
#        self.filenames = filenames4
#        self.filepaths = filepaths4
#            
#        # UPDATE DF
#        if kwargs.get("update_df", True):
#            self.df = self.df[self.df["filename"].isin(self.filenames)]
#    
    
    
    def gray_scale_finder(self, resize=0.25, write=False):
        self.gray_scale_list = []
        for filepath, filename in zip(self.filepaths, self.filenames):
            img = cv2.imread(filepath,0)
            if resize:
                img = cv2.resize(img, (0,0), fx=1*resize, fy=1*resize) 
            vec = np.ravel(img)
            mc = Counter(vec).most_common(9)
            g = [item[0] for item in mc]
            med = int(np.median(g))
            self.gray_scale_list.append(med)
            print(filename + ": " + str(med))
        print("\nMean grayscale in directory: " + str(int(np.mean(self.gray_scale_list))))
        if write == True:
            self.df["gray_scale"] = self.gray_scale_list
    
    
    def save(self, **kwargs):
        output = kwargs.get("save_dir",self.save_dir) # "out") #
        if "append" in kwargs:
            app = '_' + kwargs.get('append')
        else:
            app = ""
        path=os.path.join(output , self.name +  app + ".txt")
        if kwargs.get('overwrite',True) == False:
            if not os.path.exists(path):
                self.df.astype(str).to_csv(path_or_buf=path, sep="\t", index=False, float_format = '%.12g')
        else:
                self.df.astype(str).to_csv(path_or_buf=path, sep="\t", index=False, float_format = '%.12g')


class scale_maker:
    def __init__(self):

        # initialize -----
        self.done_step1 = False 
        self.done_step2 = False
        self.current = (0, 0) 
        self.points_step1 = [] 
        self.points_step2 = []
        self.scale_px = 0
        
    def on_mouse_step1(self, event, x, y, buttons, user_param):
        if self.done_step1: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point #%d to scale outline" % (len(self.points_step1)+1))
            self.points_step1.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points_step1) > 0:
                self.points_step1 = self.points_step1[:-1]
                print("Removing point #%d from scale outline" % (len(self.points_step1)+1))


    def on_mouse_step2(self, event, x, y, buttons, user_param):
        if self.done_step2: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point %s of 2 to scale" % (len(self.points_step2)+1))
            self.points_step2.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points_step2) > 0:
                self.points_step2 = self.points_step2[:-1]
                print("Removing point %s of 2 from scale" % (len(self.points_step2)+1))
        if len(self.points_step2)==2:
            self.done_step2 = True
            self.scale_px = int(math.sqrt( ((self.points_step2[0][0]-self.points_step2[1][0])**2)+((self.points_step2[0][1]-self.points_step2[1][1])**2)))
            cv2.destroyWindow("phenopype")


    def grab(self, image_path, length, **kwargs): 
        
        # initialize # ----------------
        image = cv2.imread(image_path)
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        unit = kwargs.get('unit', "mm")
        mode = kwargs.get('mode', "box")
        zoom = kwargs.get('zoom', False)
        # crop = kwargs.get('crop', True)
        
#        if "resize" in kwargs:
#            factor = kwargs.get('resize', 0.5)
#            image = cv2.resize(image, (0,0), fx=1*factor, fy=1*factor) 

        # =============================================================================
        # Step 1 - draw scale outline
        # =============================================================================

        print("\nMark the outline of the scale by left clicking, remove points by right clicking, finish with enter.")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self.on_mouse_step1)
        temp_canvas_1 = copy.deepcopy(image)
        
        if mode == "polygon":
            while(not self.done_step1):
                if (len(self.points_step1) > 0):
                    cv2.polylines(temp_canvas_1, np.array([self.points_step1]), False, green, 2)
                    cv2.line(temp_canvas_1, self.points_step1[-1], self.current, blue, 2)
                cv2.imshow("phenopype", temp_canvas_1)
                temp_canvas_1 = copy.deepcopy(image)
                if cv2.waitKey(50) & 0xff == 13:
                    self.done_step1 = True
                    cv2.destroyWindow("phenopype")
                elif cv2.waitKey(50) & 0xff == 27:
                    cv2.destroyWindow("phenopype")
                    break
                    sys.exit("phenopype process stopped")     
            print("Finished, scale outline drawn. Now add the scale by clicking on two points with a known distance between them:")
                
        elif mode == "box":
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            (x,y,w,h) = cv2.selectROI("phenopype", image, fromCenter=False)
            if any([cv2.waitKey(50) & 0xff == 27, cv2.waitKey(50) & 0xff == 13]):
                cv2.destroyWindow("phenopype")  
            self.points_step1 = [(x, y), (x, y+h), (x+w, y+h), (x+w, y)]
            print("Finished, scale outline drawn. Now dd the scale by clicking on two points with a known distance between them:")

        # create colour mask to show scale outline
        colour_mask = np.zeros(image.shape, np.uint8)
        colour_mask[:,:,1] = 255 # all area green
        cv2.fillPoly(colour_mask, np.array([self.points_step1]), red) # red = excluded area
        temp_canvas_1 = cv2.addWeighted(copy.deepcopy(image), .7, colour_mask, 0.3, 0) # combine
        self.image_overlay = temp_canvas_1


        # create template image for SIFT
        rx,ry,w,h = cv2.boundingRect(np.array(self.points_step1, dtype=np.int32))
        self.image_original_template = image[ry:ry+h,rx:rx+w]
        
        # create mask for SIFT
        self.mask_original_template = np.zeros(image.shape[0:2], np.uint8)
        cv2.fillPoly(self.mask_original_template, np.array([self.points_step1]), white) 
        self.mask_original_template = self.mask_original_template[ry:ry+h,rx:rx+w]
        
        # zoom into drawn scale outline for better visibility
        if zoom==True:
            temp_canvas_1 = image[ry:ry+h,rx:rx+w]
            self.done_step1 = True

        # =============================================================================
        # Step 2 - measure scale length
        # =============================================================================
        
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        temp_canvas_2 = copy.deepcopy(temp_canvas_1)
        cv2.setMouseCallback("phenopype", self.on_mouse_step2)
        
        while(not self.done_step2):
            if (len(self.points_step2) > 0):
                cv2.polylines(temp_canvas_2, np.array([self.points_step2]), False, green, 2)
                cv2.line(temp_canvas_2, self.points_step2[-1], self.current, blue, 2)
            cv2.imshow("phenopype", temp_canvas_2)
            temp_canvas_2 = copy.deepcopy(temp_canvas_1)
            if any([cv2.waitKey(50) & 0xff == 27, cv2.waitKey(50) & 0xff == 13]):
                cv2.destroyWindow("phenopype")
                break    
            
        print("\n")
        print("------------------------------------------------")
        print("Finished - your scale has %s pixel per %s %s." % (self.scale_px, length, unit))
        print("------------------------------------------------")
        print("\n")

        if kwargs.get('show', True):
            cv2.polylines(temp_canvas_2, np.array([self.points_step2]), False, red, 4)    
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("phenopype", temp_canvas_2)    
            if any([cv2.waitKey(0) & 0xff == 27, cv2.waitKey(0) & 0xff == 13]):
                cv2.destroyWindow("phenopype")
              
        self.image_zoomed = temp_canvas_2


        # SCALE
        self.measured = self.scale_px/length
        self.current = self.measured

        # REFERENCE
        (x,y),radius = cv2.minEnclosingCircle(np.array(self.points_step1))
        self.ref = (radius * 2)
        
        # MASK
        zeros = np.zeros(image.shape[0:2], np.uint8)
        self.mask = cv2.fillPoly(zeros, [np.array(self.points_step1, dtype=np.int32)], white)


    def detect(self, image_path, **kwargs):
        
        # initialize ------------------
        self.image_target = cv2.imread(image_path)
        image_target = self.image_target 
        image_original = self.image_original_template
        
        show = kwargs.get('show', False)
        min_matches = kwargs.get('min_matches', 10)

        if (image_target.shape[0] + image_target.shape[1])/2 > 2000:
            factor = kwargs.get('resize', 0.5)
            image_target = cv2.resize(image_target, (0,0), fx=1*factor, fy=1*factor) 
        
        if not len(image_target.shape)==3:
            image_target = cv2.cvtColor(image_target, cv2.COLOR_GRAY2BGR)
            
    
        # =============================================================================
        # SIFT detector
        # =============================================================================
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp1, des1 = sift.detectAndCompute(img1,self.mask_original_template)
        # kp2, des2 = sift.detectAndCompute(img2,None)
         
        # =============================================================================
        # ORB detector
        # =============================================================================
#        orb = cv2.ORB_create()
#        kp1, des1 = orb.detectAndCompute(img1,self.mask_original_template)
#        kp2, des2 = orb.detectAndCompute(img2,None)
#        des1 = np.asarray(des1, np.float32)
#       des2 = np.asarray(des2, np.float32)
        
#        FLANN_INDEX_KDTREE = 0
#        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#        search_params = dict(checks = 50)
#        flann = cv2.FlannBasedMatcher(index_params, search_params)
#        matches = flann.knnMatch(des1,des2,k=2)
        
        # =============================================================================
        # AKAZE detector
        # =============================================================================
        
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(image_original,self.mask_original_template)
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
            ret, contours, hierarchy = cv2.findContours(self.mask_original_template,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_TC89_L1)
            box = contours[0].astype(np.float32)

            rect  = cv2.perspectiveTransform(box,M).astype(np.int32)
            image_target = cv2.polylines(image_target,[rect],True,red,5, cv2.LINE_AA)
            
            # =============================================================================
            # compare scale to original, and return adjusted ratios
            # =============================================================================
            
            if show == True:
                cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
                cv2.imshow("phenopype", image_target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if kwargs.get("convert",True) == True:
                rect = rect/factor
                
            # SCALE
            rect = np.array(rect, dtype=np.int32)
            (x,y),radius = cv2.minEnclosingCircle(rect)
            self.current = round(self.measured * ((radius * 2)/self.ref),1)
            
            # MASK
            zeros = np.zeros(self.image_target.shape[0:2], np.uint8)
            self.mask = cv2.fillPoly(zeros, [np.array(rect, dtype=np.int32)], white)

            # TARGET SNIPPET
            (rx,ry,w,h) = cv2.boundingRect(rect)
            self.image_found = self.image_target[ry:ry+h,rx:rx+w]

            
            print("\n")
            print("--------------------------------------")
            print("Scale found with %d keypoint matches" % self.nkp)
            print("--------------------------------------")
            print("\n")

            return self.current, self.mask      
        
        else:
            print("\n")
            print("----------------------------------------------")
            print("Scale not found - only %d/%d keypoint matches" % (self.nkp, min_matches))
            print("----------------------------------------------")
            print("\n")
            
            return "no current scale", "no scale mask"

        
class polygon_maker:
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
                
    def draw(self, image_path,**kwargs):
        image = cv2.imread(image_path)
        mode = kwargs.get("mode","box")
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print("\nMark the outline of your arena, i.e. what you want to include in the image analysis by left clicking, finish with enter.")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self.on_mouse)
        temp_canvas = copy.deepcopy(image)
        if mode == "polygon":
            # =============================================================================
            # draw polygon outline
            # =============================================================================
            while(not self.done):
                if (len(self.points) > 0):
                    cv2.polylines(temp_canvas, np.array([self.points]), False, green, 3)
                    cv2.line(temp_canvas, self.points[-1], self.current, blue, 3)
                cv2.imshow("phenopype", temp_canvas)
                temp_canvas = copy.deepcopy(image)
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
                            
        zeros = np.zeros(image.shape[0:2], np.uint8)
        zeros.fill(255)
        self.mask = cv2.fillPoly(zeros, np.array([self.points]), black)
        boo = np.array(self.mask, dtype=bool)
        temp_canvas[boo,2] =255
        self.drawn = temp_canvas
        
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

class object_finder:
    def run(self,im_path, scale=1, **kwargs):
        """ find objects in image via thresholding
        Parameters
        ----------
        image: array_like
            image that contains your objects
        mode: str("single", "multi")
            detection of single largest object or multiple

        Parameters (optional)
        ---------------------
        blur_kern: int
            kernel size, bigger kernels remove small structures (default: 99)
        corr_factor: int
            factor (in px) to add to (positive int) or subtract from (negative int) object (default: 0)
        gray_value_ref: int (0-255)
            reference gray scale value to adjust the given picture's histogram to
        iterations: int
            needs to be specified if 'adaptive' thresholding is used (default: 3)
        min_diam: int
            minimum diameter (longest distance in contour) in pixels for objects to be included (default: 0)
        min_area: int
            minimum contour area in pixels for objects to be included (default: 0)
        resize: in (0-1)
            resize image to speed up detection process(WARNING: may result in poor detection results)
        sensitivity: int (odd)
            needs to be specified if 'adaptive' thresholding is used (default: 99)
        thresholding: {'binary', 'adaptive', 'otsu'}
            type of thresholding (default: binary)
        thresh_val: int (0-255)
            needs to be specified if 'binary' thresholding is used (default: 127)


        Returns (optional)
        ------------------
        image: array_like
            original input image (for further referencing)
        gray: array_like
            input image as grayscale
        blurred: array_like
            blurred grayscale image
        thresh: array_like (binary)
            thresholded grayscale image
        contours: list
            list of contours (lists)
        """
        # =============================================================================
        # INITIALIZE
        # =============================================================================
        
        # LOAD 
        self.mode =  kwargs.get('mode', "multi")
        self.image = cv2.imread(im_path)
        self.image_copy = self.image
        resize = kwargs.get("resize", 1)
        self.image = cv2.resize(self.image, (0,0), fx=1*resize, fy=1*resize) 
        self.filename = os.path.basename(im_path)
        
        # GET IMAGE DATE
        try:
            self.date_taken = exif_date(im_path)
        except:
            self.date_taken = "NA"
        self.date_analyzed = str(datetime.datetime.now())[:19]
            
        # APPLY GRAY-CORRECTION FACTOR TO GRAYSCALE IMAGE AND ROI
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        if 'gray_value_ref' in kwargs:
            if resize < 1:
                ret = gray_scale(source=self.gray)
            else: 
                ret = gray_scale(source=self.gray,  resize=0.25)
            ref = kwargs.get('gray_value_ref', ret)
            self.gray_corr_factor = int(ref - ret)
            self.gray_corrected = np.array(copy.deepcopy(self.gray) + self.gray_corr_factor, dtype="uint8")
            self.drawn = self.gray_corrected
        else:
             self.drawn = copy.deepcopy(self.gray)     
             
        self.drawn1 = self.drawn

        self.drawn = cv2.cvtColor(self.drawn,cv2.COLOR_GRAY2BGR)
        

        # =============================================================================
        # BLUR1 > THRESHOLDING > MORPHOLOGY > BLUR2
        # =============================================================================
        
        # BLUR 1ST PASS
        if "blur1" in kwargs:
            blur_kernel = kwargs.get("blur1", 1)
            self.blurred = blur(self.gray, blur_kernel)
        else:
            self.blurred = self.gray
            
        # THRESHOLDING   
        method = kwargs.get('method', ("otsu",""))
        thresholding = method[0]
        
        if thresholding == "otsu":
            ret, self.thresh = cv2.threshold(self.blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif thresholding == "adaptive":
            self.thresh = cv2.adaptiveThreshold(self.blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,method[1], method[2])
        elif thresholding == "binary":
            ret, self.thresh = cv2.threshold(self.blurred,method[1], 255,cv2.THRESH_BINARY_INV)
        self.morph = self.thresh
                   
        # BLUR 2ND PASS
        if "blur2" in kwargs:
            blur_kernel, thresh_val = kwargs.get("blur2")
            self.morph = blur(self.morph, blur_kernel)
            ret, self.morph = cv2.threshold(self.morph, thresh_val, 255,cv2.THRESH_BINARY)

            
        # BORDER CORRECTION FACTOR
        if "corr_factor" in kwargs:
            corr_factor = kwargs.get("corr_factor")
            size = abs(corr_factor[0])
            iterations = corr_factor[1]
            
            if corr_factor[2] == "cross":
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(size,size))
            elif corr_factor[2] == "ellipse":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
            
            corr_factor = kwargs.get('corr_factor', 0)
            if corr_factor[0] < 0:
                self.morph = cv2.erode(self.morph, kernel, iterations = iterations)
            if corr_factor[0] > 0:
                self.morph = cv2.dilate(self.morph, kernel, iterations = iterations)

    
        # APPLY ARENA MASK
        if "exclude" in kwargs:
            self.mask = sum(kwargs.get('exclude'))
            self.mask = cv2.resize(self.mask, (0,0), fx=1*resize, fy=1*resize) 
            self.morph = cv2.subtract(self.morph,self.mask)
            self.morph[self.morph==1] = 0

        # =============================================================================
        # MULTI-MODE
        # =============================================================================

        if self.mode == "multi":
            idx = 0
            idx_noise = 0
            length_idx = 0
            area_idx = 0
            df_list = []
            
            label = kwargs.get("label", True)
            
            self.roi_bgr_list = []
            self.roi_gray_list = []
            self.roi_mask_list = []
            
            ret, self.contours, hierarchy = cv2.findContours(self.morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
            
            # LOOP THROUGH CONTOURS 
            if self.contours:
                for cnt in self.contours:
                    if len(cnt) > 5:
                        
                        # SIZE / AREA CHECK
                        (x,y),radius = cv2.minEnclosingCircle(cnt)
                        x, y= int(x), int(y)
                        length = int(radius * 2)
                        area = int(cv2.contourArea(cnt))
                        cont = True
                        if not length > kwargs.get('min_diam', 0):
                            length_idx += 1
                            cont = False
                        if not area > kwargs.get('min_area', 0):
                            area_idx += 1
                            cont=False
                        if not cont == False:
                            idx += 1
                            rx,ry,rw,rh = cv2.boundingRect(cnt)
                            
                            # GET VALUES FROM MASKED ROI
                            grayscale =  ma.array(data=self.gray[ry:ry+rh,rx:rx+rw], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                            b =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,0], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                            g =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,1], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                            r =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,2], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                            mean1 = int(np.mean(grayscale)) # mean grayscale value
                            sd1 = int(np.std(grayscale)) # standard deviation of grayscale values
                            bgr1 = (int(np.mean(b)),int(np.mean(g)),int(np.mean(r))) # mean grayscale value
                            bgr_sd1 = (int(np.std(b)),int(np.std(g)),int(np.std(r))) # mean grayscale value
                            df_list.append([self.filename, self.date_taken, self.date_analyzed, idx, x, y, scale, length, area, mean1, sd1, bgr1, bgr_sd1])
                            self.roi_bgr_list.append(self.image[ry:ry+rh,rx:rx+rw,])
                            self.roi_gray_list.append(self.gray[ry:ry+rh,rx:rx+rw,])
                            self.roi_mask_list.append(self.thresh[ry:ry+rh,rx:rx+rw])

                            # DRAW TO ROI
                            q=kwargs.get("roi_size",300)/2
                            if label==True:
                                cv2.putText(self.drawn,  str(idx) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),5,cv2.LINE_AA)
                            cv2.drawContours(self.drawn, [cnt], 0, blue, 4)
                    else:
                        idx_noise += 1
            else: 
                print("No objects found - change parameters?")
            


        # =============================================================================
        # SINGLE-MODE
        # =============================================================================
                
        elif self.mode =="single":
            df_list = []
            ret, self.contours, hierarchy = cv2.findContours(self.morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
            
            # LOOP THROUGH CONTOURS AND PICK LARGEST
            if self.contours:
                areas = [cv2.contourArea(cnt) for cnt in self.contours]                
                cnt = self.contours[np.argmax(areas)]
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                x, y= int(x), int(y)
                length1 = int(radius * 2)
                area1 = int(cv2.contourArea(cnt))
                if length1 > kwargs.get('min_diam', 0) and area1 > kwargs.get('min_area', 0):
                    # return contour
                    self.contour = cnt
                    
                    idx = 1
                    rx,ry,rw,rh = cv2.boundingRect(cnt)
                    
                    # GET VALUES FROM MASKED ROI
                    grayscale =  ma.array(data=self.gray[ry:ry+rh,rx:rx+rw], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                    b =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,0], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                    g =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,1], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                    r =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,2], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                    mean1 = int(np.mean(grayscale)) # mean grayscale value
                    sd1 = int(np.std(grayscale)) # standard deviation of grayscale values
                    bgr1 = (int(np.mean(b)),int(np.mean(g)),int(np.mean(r))) # mean grayscale value
                    bgr_sd1 = (int(np.std(b)),int(np.std(g)),int(np.std(r))) # mean grayscale value
                    df_list = [[self.filename, self.date_taken, self.date_analyzed, idx, x, y, scale, length1, area1, mean1, sd1, bgr1, bgr_sd1]]

                    # DRAW TO ROI
                    if "roi_size" in kwargs:
                        q=kwargs.get("roi_size",300)/2
                        cv2.rectangle(self.drawn,(int(max(0,x-q)),int(max(0, y-q))),(int(min(self.image.shape[1],x+q)),int(min(self.image.shape[0],y+q))),red,8)
                    cv2.drawContours(self.drawn, [cnt], 0, blue, int(10 * resize))
                    
                else: 
                    print("Object not bigger than minimum diameter or area")
            else: 
                print("No objects found - change parameters?")

        # =============================================================================
        # RETURN DF AND IMAGE
        # =============================================================================    

        # CREATE DF
        if len(df_list)>0:
            self.df = pd.DataFrame(data=df_list, columns = ["filename","date_taken", "date_analyzed", "idx", "x", "y", "scale","length1", "area1", "mean1", "sd1", "bgr1", "bgr_sd1"])
        else: 
            self.df = pd.DataFrame(data=[["NA"] * 13], columns = ["filename","date_taken", "date_analyzed", "idx", "x", "y", "scale","length1", "area1", "mean1", "sd1", "bgr1", "bgr_sd1"])
        self.df.set_index('filename', drop=True, inplace=True)
        
        if hasattr(self,'gray_corr_factor'):
            self.df.insert(3, "gray_corr_factor", self.gray_corr_factor)
        self.df.insert(3, "resize_factor", resize)
        
        show = kwargs.get('show', "")
        
        # SHOW DF
        if "df" in show:        

            self.df_short = self.df[["idx", "length1", "area1","mean1"]]
            self.df_short.set_index("idx", inplace=True)
            
            all_pts = len(self.contours)
            good_pts = len(self.df)
            
            
            if self.mode == "multi":
                
                print(self.df_short)
                
                print("\n")
                print("----------------------------------------------------------------")
                print("Found " + str(all_pts) + " objects in " + self.filename + ":")
                print("  ==> %d are valid objects" % good_pts)
                if not idx_noise == 0:
                    print("    - %d are noise" % idx_noise)
                if not length_idx == 0:
                    print("    - %d are not bigger than minimum diameter" % length_idx)
                if not area_idx ==0:
                    print("    - %d are not bigger than minimum area" % area_idx)
                print("----------------------------------------------------------------")
            else:
                print("----------------------------------------------------------------")
                print("Found following object in " + self.filename + ":")
                print("----------------------------------------------------------------")
                
                print(self.df_short)

        # SHOW IMAGE
        if hasattr(self,'mask'):
            boo = np.array(self.mask, dtype=bool)
            #self.drawn = copy.deepcopy(self.image)
            self.drawn[boo,2] = 255

        if "image" in show:
            cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
            cv2.imshow('phenopype', self.drawn)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        if "image_zoomed" in show:
            
            if "exclude" in kwargs:
                zoom_mask = sum(kwargs.get('exclude'))
                if "resize" in kwargs:
                    zoom_mask = cv2.resize(cv2.bitwise_not(zoom_mask), (0,0), fx=1*resize, fy=1*resize)
                else:
                    zoom_mask = cv2.bitwise_not(zoom_mask)
                ret, contours, hierarchy = cv2.findContours(zoom_mask,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_TC89_L1)
                rx,ry,rw,rh = cv2.boundingRect(contours[0])
                self.zoomed = self.drawn[ry:ry+rh,rx:rx+rw]
            else:
                self.zoomed = self.drawn
            
            cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
            cv2.imshow('phenopype', self.zoomed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                        
                        # GET VALUES FROM MASKED ROI

    def save(self, save_to, **kwargs):
        
        # set dir and names
        out_dir = save_to     
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        if "append" in kwargs:
            app = '_' + kwargs.get('append')
        else:
            app = ""
            
        filename = kwargs.get('filename',self.filename)      
        name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        
        im_path=os.path.join(out_dir , name +  app + ext)
        df_path=os.path.join(out_dir , name +  app + ".txt")
             
        # image
        image = kwargs.get("image", self.drawn)
        if kwargs.get('overwrite',True) == False:
            if not os.path.exists(im_path):
                cv2.imwrite(im_path, image)
        else:
            cv2.imwrite(im_path, image)

        # df
        df = kwargs.get("df", self.df)
        df = df.fillna(-9999)
        df = df.astype(str)
        if kwargs.get('overwrite',True) == False:
            if not os.path.exists(df_path):
                df.to_csv(path_or_buf=df_path, sep="\t", index=False)
        else:
                df.to_csv(path_or_buf=df_path, sep="\t", index=False)
                    
                    




