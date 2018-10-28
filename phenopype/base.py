# -*- coding: utf-8 -*-
"""
Last Update: 2018/10/02
Version 0.4.0
@author: Moritz Lürig
"""

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import copy
import math

import cv2
import datetime
from collections import Counter

from phenopype.utils import (red, green, blue, black, white)
from phenopype.utils import (blur, exif_date, gray_scale)

#%%
class project:
    def __init__(self):
        self.name = "phenopype project"
        
    def project_maker(self, name, files, mode="walk", **kwargs):

        # initialize
        self.name = name
        self.files = files
        self.output = kwargs.get("output",os.path.join(os.getcwd(),"output")) 
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.mode = mode
        self.filepaths = []
        self.filenames = []

        if "filetype" in kwargs:
            self.file_type = kwargs.get("filetype")
        if "exclude" in kwargs:
            self.exclude = kwargs.get("exclude")
            
        self.resize = kwargs.get("resize",1)

        # =============================================================================
        # go through directory and make filelist
        # =============================================================================
        
        if self.mode == "walk":
            for root, dirs, files in os.walk(self.files):
                for i in os.listdir(root):
                    path = os.path.join(root,i)
                    if os.path.isfile(path):
                        if hasattr(self,'file_type'):
                            if self.in_dir.endswith(self.file_type):
                                if hasattr(self,'exclude'):
                                    if not any([j in i for j in self.exclude]):
                                        self.filepaths.append(path)
                                        self.filenames.append(i)
                                else:
                                    self.filepaths.append(path)
                                    self.filenames.append(i)
                        else: 
                            if hasattr(self,'exclude'):
                                if not any([j in i for j in self.exclude]):
                                    self.filepaths.append(path)
                                    self.filenames.append(i)
                            else:
                                self.filepaths.append(path)
                                self.filenames.append(i)
                                
        elif self.mode == "dir":
            for i in os.listdir(self.files):
                path = os.path.join(self.files,i)
                if os.path.isfile(path):
                    if hasattr(self,'file_type'):
                        if self.in_dir.endswith(self.file_type):
                            if hasattr(self,'exclude'):
                                if not any([j in i for j in self.exclude]):
                                    self.filepaths.append(path)
                                    self.filenames.append(i)
                            else:
                                self.filepaths.append(path)
                                self.filenames.append(i)
                    else: 
                        if hasattr(self,'exclude'):
                            if not any([j in i for j in self.exclude]):
                                self.filepaths.append(path)
                                self.filenames.append(i)
                        else:
                            self.filepaths.append(path)
                            self.filenames.append(i)
                    
        elif self.mode == "single":
            self.filepaths.append(self.files)
            self.filenames.append(os.path.basename(self.files))

        # =============================================================================
        # make project dataframe
        # =============================================================================
        self.df = pd.DataFrame(data=list(zip(self.filepaths, self.filenames)), columns = ["filepath", "filename"])
        self.df.index = self.filenames
        self.df.insert(0, "project", self.name)
        self.df.drop_duplicates(subset="filename", inplace=True)
        self.filepaths = self.df['filepath'].tolist()
        self.filenames = self.df['filename'].tolist()
        self.df.drop(columns='filepath', inplace=True)

    def update_list(self):
        self.filepaths = []
        self.filenames = []
        if self.mode == "walk":
            for root, dirs, files in os.walk(self.files):
                for i in os.listdir(root):
                    path = os.path.join(root,i)
                    if os.path.isfile(path):
                        if hasattr(self,'file_type'):
                            if self.in_dir.endswith(self.file_type):
                                if hasattr(self,'exclude'):
                                    if not any([j in i for j in self.exclude]):
                                        self.filepaths.append(path)
                                        self.filenames.append(i)
                                else:
                                    self.filepaths.append(path)
                                    self.filenames.append(i)
                        else: 
                            if hasattr(self,'exclude'):
                                if not any([j in i for j in self.exclude]):
                                    self.filepaths.append(path)
                                    self.filenames.append(i)
                            else:
                                self.filepaths.append(path)
                                self.filenames.append(i)
                                
        elif self.mode == "dir":
            for i in os.listdir(self.files):
                path = os.path.join(self.files,i)
                if os.path.isfile(path):
                    if hasattr(self,'file_type'):
                        if self.in_dir.endswith(self.file_type):
                            if hasattr(self,'exclude'):
                                if not any([j in i for j in self.exclude]):
                                    self.filepaths.append(path)
                                    self.filenames.append(i)
                            else:
                                self.filepaths.append(path)
                                self.filenames.append(i)
                    else: 
                        if hasattr(self,'exclude'):
                            if not any([j in i for j in self.exclude]):
                                self.filepaths.append(path)
                                self.filenames.append(i)
                        else:
                            self.filepaths.append(path)
                            self.filenames.append(i)
                    
        elif self.mode == "single":
            self.filepaths.append(self.files)
            self.filenames.append(os.path.basename(self.files))
            
            
        self.df = self.df[self.df["filename"].isin(self.filenames)]



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
        output = kwargs.get("output",self.output) # "out") #
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

#%%
class scale_maker:
    def __init__(self):

        # setting up temporary variables
        self.window_name = "phenopype" # Name for our window
        self.done_step1 = False 
        self.done_step2 = False
        self.current = (0, 0) 
        self.points_step1 = [] # List of points defining our polygon
        self.points_step2 = [] # List of points defining our polygon

    def on_mouse_step1(self, event, x, y, buttons, user_param):
        if self.done_step1: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point #%d to scale outline" % (len(self.points_step1)+1))
            self.points_step1.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            self.done_step1 = True

    def on_mouse_step2(self, event, x, y, buttons, user_param):
        if self.done_step2: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point %s of 2 to scale" % (len(self.points_step2)+1))
            self.points_step2.append((x, y))
        if len(self.points_step2)==2:
            self.done_step2 = True
            self.scale_px = int(math.sqrt( ((self.points_step2[0][0]-self.points_step2[1][0])**2)+((self.points_step2[0][1]-self.points_step2[1][1])**2)))

    def measure(self, im_path, **kwargs): #, 
        # initialize # ----------------
        length = kwargs.get('length', 10)
        unit = kwargs.get('unit', "mm")
        zoom = kwargs.get('zoom', False)
        crop = kwargs.get('crop', True)
        image = cv2.imread(im_path)
        if "resize" in kwargs:
            factor = kwargs.get('resize', 0.5)
            image = cv2.resize(image, (0,0), fx=1*factor, fy=1*factor) 

        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # =============================================================================
        # Step 1 - draw scale outline
        # =============================================================================
        if crop == True:
            print("\nMark the outline of the scale by left clicking, finish by right clicking:")
            cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.on_mouse_step1)
            temp_canvas_1 = copy.deepcopy(image)
    
            while(not self.done_step1):
                if (len(self.points_step1) > 0):
                    cv2.polylines(temp_canvas_1, np.array([self.points_step1]), False, green, 2)
                    cv2.line(temp_canvas_1, self.points_step1[-1], self.current, blue, 2)
                cv2.imshow(self.window_name, temp_canvas_1)
                temp_canvas_1 = copy.deepcopy(image)
                if (cv2.waitKey(50) & 0xff) == 27:
                    cv2.destroyWindow(self.window_name)
                    break

            cv2.destroyWindow(self.window_name)
            
            print("Finished, scale outline drawn. Add the scale by clicking on two points with a known distance between them:")

            # create colour mask to show scale outline
            colour_mask = np.zeros(image.shape, np.uint8)
            colour_mask[:,:,1] = 255 # all area green
            cv2.fillPoly(colour_mask, np.array([self.points_step1]), red) # red = excluded area
            temp_canvas_1 = cv2.addWeighted(copy.deepcopy(image), .7, colour_mask, 0.3, 0) # combine
            
            # create image for SIFT
            rx,ry,w,h = cv2.boundingRect(np.array(self.points_step1, dtype=np.int32))
            self.image = image[ry:ry+h,rx:rx+w]
            # create mask for SIFT
            self.mask_det = np.zeros(image.shape[0:2], np.uint8)
            cv2.fillPoly(self.mask_det, np.array([self.points_step1]), white) 
            self.mask_det = self.mask_det[ry:ry+h,rx:rx+w]
            
            # zoom into drawn scale outline for better visibility
            if zoom==True:
                # colour mask for step 2
                temp_canvas_1 = temp_canvas_1[ry:ry+h,rx:rx+w]

        else:
            self.done_step1 = True
            print("Add the scale by clicking on two points with a known distance between them:")
            temp_canvas_1 = image
        # =============================================================================
        # Step 2 - measure scale length
        # =============================================================================
        
        temp_canvas_2 = copy.deepcopy(temp_canvas_1)
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse_step2)

        while(not self.done_step2):
            if (len(self.points_step2) > 0):
                cv2.polylines(temp_canvas_2, np.array([self.points_step2]), False, green, 2)
                cv2.line(temp_canvas_2, self.points_step2[-1], self.current, blue, 2)
            cv2.imshow(self.window_name, temp_canvas_2)
            temp_canvas_2 = copy.deepcopy(temp_canvas_1)
            if (cv2.waitKey(50) & 0xff) == 27:
                cv2.destroyWindow(self.window_name)
                break
            
        print("Finished, scale drawn. your scale has %s pixel per %s %s." % (self.scale_px, length, unit))

        # =============================================================================
        # give per pixel ratios
        # =============================================================================
        cv2.polylines(temp_canvas_2, np.array([self.points_step2]), False, black, 2)
        
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, temp_canvas_2)
        if (cv2.waitKey(0) & 0xff) == 27:
            cv2.destroyWindow(self.window_name)

        self.measured = self.scale_px/length
        self.current = self.measured
        
        if crop == True:
            (x,y),radius = cv2.minEnclosingCircle(np.array(self.points_step1))
            self.ref = (radius * 2)
            zeros = np.zeros(image.shape[0:2], np.uint8)
            self.mask = cv2.fillPoly(zeros, [np.array(self.points_step1, dtype=np.int32)], white)


    def find(self, im_path, **kwargs):
        # initialize ------------------
        self.image_current = cv2.imread(im_path)
        factor = kwargs.get('resize', 0.5)
        image = cv2.resize(copy.deepcopy(self.image_current), (0,0), fx=1*factor, fy=1*factor) 
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        min_matches = kwargs.get('min_keypoints', 10)
        show = kwargs.get('show', False)
        img1 = self.image
        img2 = copy.deepcopy(image)

        # =============================================================================
        # SIFT detector
        # =============================================================================
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,self.mask_det)
        kp2, des2 = sift.detectAndCompute(img2,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>min_matches:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            ret, contours, hierarchy = cv2.findContours(self.mask_det,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_TC89_L1)
            box = contours[0].astype(np.float32)
            rect  = cv2.perspectiveTransform(box,M).astype(np.int32)
            img2 = cv2.polylines(img2,[rect],True,red,5, cv2.LINE_AA)
    
            # =============================================================================
            # compare scale to original, and return adjusted ratios
            # =============================================================================
            if show == True:
                cv2.namedWindow("window", flags=cv2.WINDOW_NORMAL)
                cv2.imshow("window", img2)
            if kwargs.get("convert",True) == True:
                rect = rect/factor
            rect = np.array(rect, dtype=np.int32)
            (x,y),radius = cv2.minEnclosingCircle(rect)
            self.current = round(self.measured * ((radius * 2)/self.ref),1)
            zeros = np.zeros(self.image_current.shape[0:2], np.uint8)
            self.mask = cv2.fillPoly(zeros, [np.array(rect, dtype=np.int32)], white)

            return self.current, self.mask

        else:
            print("Scale not found - Only %d/%d matches" % (len(good),min_matches))
        


#%%
class polygon_maker:
    def __init__(self):

        # initialize # ----------------
        self.window_name = "phenopype" # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point #%d with position(%d,%d) to arena" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Completing arena with %d points." % len(self.points))
            self.done = True

    def draw(self, im_path, **kwargs):
        image = cv2.imread(im_path)
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print("\nMark the outline of your arena, i.e. what you want to include in the image analysis by left clicking, finish by right clicking:")
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        temp_canvas = copy.deepcopy(image)

        # =============================================================================
        # draw polygon outline
        # =============================================================================
        while(not self.done):
            if (len(self.points) > 0):
                cv2.polylines(temp_canvas, np.array([self.points]), False, green, 3)
                cv2.line(temp_canvas, self.points[-1], self.current, blue, 3)
            cv2.imshow(self.window_name, temp_canvas)
            temp_canvas = copy.deepcopy(image)
            if (cv2.waitKey(50) & 0xff) == 27:
                cv2.destroyWindow(self.window_name)
                break
            
        zeros = np.zeros(image.shape[0:2], np.uint8)
        zeros.fill(255)
        self.mask = cv2.fillPoly(zeros, np.array([self.points]), black)

        if kwargs.get('show', False) == True:
            boo = np.array(self.mask, dtype=bool)
            temp_canvas[boo,2] =255
            cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, temp_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(1)
            cv2.destroyAllWindows()

#        cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
#        cv2.imshow('phenopype',  img)

#%%

class standard_object_finder:
    def run(self,im_path, scale, **kwargs):
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
        min_size: int
            minimum size (longest distance in contour) in pixels for objects to be included (default: 0)
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
            self.drawn = copy.deepcopy(self.gray) + self.gray_corr_factor
        else:
             self.drawn = copy.deepcopy(self.gray)            
        self.drawn = np.array(self.drawn, dtype="uint8")
        self.drawn = cv2.cvtColor(self.drawn,cv2.COLOR_GRAY2BGR)


        # =============================================================================
        # BLUR, THRESHOLDING, MASK
        # =============================================================================
        
        # BLUR BEFORE
        blur_kernel = kwargs.get("blur1", 1)
        self.blurred = blur(self.gray, blur_kernel)
            
        # THRESHOLDING   
        thresholding = kwargs.get('thresholding', "otsu")
        if thresholding == "otsu":
            ret, self.thresh = cv2.threshold(self.blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif thresholding == "adaptive":
            self.thresh = cv2.adaptiveThreshold(self.gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,kwargs.get('sensitivity', 99), kwargs.get('iterations', 3))
        elif thresholding == "binary":
            ret, self.thresh = cv2.threshold(self.blurred,kwargs.get('bin_thresh', 127), 255,cv2.THRESH_BINARY_INV)
            
        # BORDER CORRECTION FACTOR
        corr_factor = kwargs.get('corr_factor', 0)
        if corr_factor < 0:
            corr_factor = abs(corr_factor)
            self.thresh = cv2.erode(self.thresh,np.ones((corr_factor,corr_factor),np.uint8), iterations = 1)
        if corr_factor > 0:
            self.thresh = cv2.dilate(self.thresh,np.ones((corr_factor,corr_factor),np.uint8), iterations = 1)
            
        # BLUR AFTER
        if "blur2" in kwargs:
            blur_kernel, thresh_val = kwargs.get("blur2")
            self.thresh = blur(self.thresh, blur_kernel)
            ret, self.morph = cv2.threshold(self.blurred, thresh_val, 255,cv2.THRESH_BINARY)
        else:
            self.thresh = self.thresh

        # APPLY ARENA MASK
        if "exclude" in kwargs:
            self.mask = sum(kwargs.get('exclude'))
            self.mask = cv2.resize(self.mask, (0,0), fx=1*resize, fy=1*resize) 
            self.thresh = cv2.subtract(self.thresh,self.mask)
            self.thresh[self.thresh==1] = 0


        # =============================================================================
        # MULTI-MODE
        # =============================================================================

        if self.mode == "multi":
            idx = 0
            self.df_list = []
            ret, self.contours, hierarchy = cv2.findContours(self.thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
            
            # LOOP THROUGH CONTOURS 
            if self.contours:
                for cnt in self.contours:
                    if len(cnt) > 5:
                        (x,y),radius = cv2.minEnclosingCircle(cnt)
                        x, y= int(x), int(y)
                        length = int(radius * 2)
                        area = int(cv2.contourArea(cnt))
                        if length > kwargs.get('min_size', 0) and area > kwargs.get('min_area', 0):
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
                            self.df_list.append([self.filename, self.date_taken, self.date_analyzed, idx, x, y, scale, length, area, mean1, sd1, bgr1, bgr_sd1])
                            
                            # DRAW TO ROI
                            if "roi_size" in kwargs:
                                q=kwargs.get("roi_size",300)/2
                                cv2.rectangle(self.drawn,(int(max(0,x-q)),int(max(0, y-q))),(int(min(self.image.shape[1],x+q)),int(min(self.image.shape[0],y+q))),red,8)
                            if "label" in kwargs:
                                cv2.putText(self.drawn,  str(idx) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),5,cv2.LINE_AA)
                            cv2.drawContours(self.drawn, [cnt], 0, green, 4)
                    else: 
                        print("Object not bigger than min_size or min_area")
            else: 
                print("No objects found - change parameters?")


        # =============================================================================
        # SINGLE-MODE
        # =============================================================================
        
        elif self.mode =="single":
            self.df_list = []
            ret, self.contours, hierarchy = cv2.findContours(self.thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
            
            # LOOP THROUGH CONTOURS AND PICK LARGEST
            if self.contours:
                areas = [cv2.contourArea(cnt) for cnt in self.contours]                
                cnt = self.contours[np.argmax(areas)]
                if len(cnt) > 5:
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    x, y= int(x), int(y)
                    length1 = int(radius * 2)
                    area1 = int(cv2.contourArea(cnt))
                    if length1 > kwargs.get('min_size', 0) and area1 > kwargs.get('min_area', 0):
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
                        self.df_list = [[self.filename, self.date_taken, self.date_analyzed, idx, x, y, scale, length1, area1, mean1, sd1, bgr1, bgr_sd1]]

                        # DRAW TO ROI
                        if "roi_size" in kwargs:
                            q=kwargs.get("roi_size",300)/2
                            cv2.rectangle(self.drawn,(int(max(0,x-q)),int(max(0, y-q))),(int(min(self.image.shape[1],x+q)),int(min(self.image.shape[0],y+q))),red,8)
                        cv2.drawContours(self.drawn, [cnt], 0, blue, int(10 * resize))
                        
                    else: 
                        print("Object not bigger than min_size or min_area")
            else: 
                print("No objects found - change parameters?")
    
        # =============================================================================
        # RETURN DF AND IMAGE
        # =============================================================================    

        # DATAFRAME
        if len(self.df_list)>0:
            self.df = pd.DataFrame(data=self.df_list, columns = ["filename","date_taken", "date_analyzed", "idx", "x", "y", "scale","length1", "area1", "mean1", "sd1", "bgr1", "bgr_sd1"])
        else: 
            self.df = pd.DataFrame(data=[["NA"] * 13], columns = ["filename","date_taken", "date_analyzed", "idx", "x", "y", "scale","length1", "area1", "mean1", "sd1", "bgr1", "bgr_sd1"])
        self.df.set_index('filename', drop=True, inplace=True)
        
        if hasattr(self,'gray_corr_factor'):
            self.df.insert(3, "gray_corr_factor", self.gray_corr_factor)
        self.df.insert(3, "resize_factor", resize)

        if kwargs.get('show_df', False) == True:
            self.df_short = self.df[["scale", "length1", "area1","mean1","bgr1","gray_corr_factor"]]
            print("----------------------------------------------------------")
            print("Found " + str(len(self.df)) + " objects in " + self.filename)
            print("----------------------------------------------------------")
            print(self.df_short) # 
            
        # IMAGE
        if hasattr(self,'mask'):
            boo = np.array(self.mask, dtype=bool)
            #self.drawn = copy.deepcopy(self.image)
            self.drawn[boo,2] = 255

        if kwargs.get('show_img', False) == True:
            cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
            cv2.imshow('phenopype', self.drawn)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def save(self, **kwargs):
        # set dir and names
        output = kwargs.get("output",os.path.join(os.getcwd(),"output")) # "out") #
        if not os.path.exists(output):
            os.makedirs(output)
        if "append" in kwargs:
            app = '_' + kwargs.get('append')
        else:
            app = ""
        filename = kwargs.get('filename',self.filename)
        name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        im_path=os.path.join(output , name +  app + ext)
        df_path=os.path.join(output , name +  app + ".txt")
        
        image = kwargs.get("image")
        # image
        if "image" in kwargs:
            if kwargs.get('overwrite',True) == False:
                if not os.path.exists(im_path):
                    cv2.imwrite(im_path, image)
            else:
                cv2.imwrite(im_path, image)

        # df
        if "df" in kwargs:
            df = kwargs.get("df")
            df = df.fillna(-9999)
            df = df.astype(str)
            if kwargs.get('overwrite',True) == False:
                if not os.path.exists(df_path):
                    df.to_csv(path_or_buf=df_path, sep="\t", index=False)
            else:
                    df.to_csv(path_or_buf=df_path, sep="\t", index=False)
                    
                    




