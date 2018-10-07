# -*- coding: utf-8 -*-
"""
Last Update: 2018/10/02
Version 0.4.0
@author: Moritz Lürig
"""

import os
import cv2
import numpy as np
import numpy.ma as ma
import pandas as pd
import copy
import math

from collections import Counter

from utils import blur, exif_date, gray_scale
from utils import red, green, blue, white, black

#%%
class project:
    def __init__(self):
        self.in_dir = None
        self.out_dir = None
        self.scale = None
        self.scale_image = None
        #self.gray_scale = None
        
    def files(self, files = None, out_dir=None, mode="walk", **kwargs):
        
        # initialize
        self.files = files
        self.out_dir = out_dir
        self.mode = mode
        self.file_list = []
        if "filetype" in kwargs:
            self.file_type = kwargs.get("filetype")
        
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
                                self.file_list.append(path)
                        else: 
                            self.file_list.append(path)

        elif self.mode == "dir":
            for i in os.listdir(root):
                path = os.path.join(root,i)
                if os.path.isfile(path):
                    if hasattr(self,'file_type'):
                        if self.files.endswith(self.file_type):
                            self.file_list.append(path)
                    else: 
                        self.file_list.append(path)
                        
        elif self.mode == "single":
            self.file_list.append(self.files)


    def gray_scale(self, resize=0.25):
        self.gray_scale_list = []
        for file in self.file_list:
            img = cv2.imread(file,0)
            if resize:
                img = cv2.resize(img, (0,0), fx=1*resize, fy=1*resize) 
            vec = np.ravel(img)
            mc = Counter(vec).most_common(9)
            g = [item[0] for item in mc]
            med = int(np.median(g))
            self.gray_scale_list.append(med)
            print(os.path.basename(file) + ": " + str(med))
        print("\nMean grayscale in directory: " + str(int(np.mean(self.gray_scale_list))))

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

    def draw(self, im_path, **kwargs): #, 
        # initialize # ----------------
        length = kwargs.get('length', 10)
        unit = kwargs.get('unit', "mm")
        zoom = kwargs.get('zoom', False)
        crop = kwargs.get('zoom', True)
        image = cv2.imread(im_path)
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
                cv2.waitKey(50)
            cv2.destroyWindow(self.window_name)
        else:
            self.done_step2 = True
        print("Finished, scale outline drawn. Add the scale by clicking on two points with a known distance between them:")

        # create colour mask to show scale outline
        colour_mask = np.zeros(image.shape, np.uint8)
        colour_mask[:,:,1] = 255 # all area green
        cv2.fillPoly(colour_mask, np.array([self.points_step1]), red) # red = excluded area
        temp_canvas_1 = cv2.addWeighted(copy.deepcopy(image), .7, colour_mask, 0.3, 0) # combine

        # create image for SIFT
        rx,ry,w,h = cv2.boundingRect(np.array(self.points_step1, dtype=np.int32))
        self.img = image[ry:ry+h,rx:rx+w]
        # create mask for SIFT
        self.mask_det = np.zeros(image.shape[0:2], np.uint8)
        cv2.fillPoly(self.mask_det, np.array([self.points_step1]), white) 
        self.mask_det = self.mask_det[ry:ry+h,rx:rx+w]
        
        # zoom into drawn scale outline for better visibility
        if zoom==True:
            # colour mask for step 2
            temp_canvas_1 = temp_canvas_1[ry:ry+h,rx:rx+w]

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
            cv2.waitKey(50)
        print("Finished, scale drawn. your scale has %s pixel per %s %s." % (self.scale_px, length, unit))

        # =============================================================================
        # give per pixel ratios
        # =============================================================================
        cv2.polylines(temp_canvas_2, np.array([self.points_step2]), False, black, 2)
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, temp_canvas_2)
        self.original = self.scale_px/length
        (x,y),radius = cv2.minEnclosingCircle(np.array(self.points_step1))
        self.ref = (radius * 2)
        zeros = np.zeros(image.shape[0:2], np.uint8)
        self.mask = cv2.fillPoly(zeros, [np.array(self.points_step1, dtype=np.int32)], white)

        return self.original, self.mask



    def find(self, im_path, **kwargs):
        # initialize ------------------
        image = cv2.imread(im_path)
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        min_matches = kwargs.get('min_keypoints', 10)
        show = kwargs.get('show', False)
        img1 = self.img
        img2 = copy.deepcopy(image)
        if "resize" in kwargs:
            factor = kwargs.get('resize', 0.5)
            img2 = cv2.resize(img2, (0,0), fx=1*factor, fy=1*factor) 

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
            if "resize" in kwargs:
                rect = rect/factor
            rect = np.array(rect, dtype=np.int32)
            (x,y),radius = cv2.minEnclosingCircle(rect)
            self.current = round(self.original * ((radius * 2)/self.ref),1)
            zeros = np.zeros(image.shape[0:2], np.uint8)
            self.mask = cv2.fillPoly(zeros, [np.array(rect, dtype=np.int32)], white)

            return self.current, zeros

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
            cv2.waitKey(50)
            
        zeros = np.zeros(image.shape[0:2], np.uint8)
        zeros.fill(255)
        self.mask = cv2.fillPoly(zeros, np.array([self.points]), black)

        if kwargs.get('show', True) == True:
            boo = np.array(self.mask, dtype=bool)
            temp_canvas[boo,2] =255
            cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, temp_canvas)
            cv2.waitKey(200)
        else:
            cv2.destroyAllWindows()

#        cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
#        cv2.imshow('phenopype',  img)

#%%

class object_finder:
    def run(self,im_path, scale, **kwargs):
        """ find objects in image via thresholding
        Parameters
        ----------
        image : array_like
            image that contains your objects
            
        Parameters (optional)
        ---------------------
        blur_kern: int
            kernel size, bigger kernels remove small structures (default: 99)
        corr_factor: int
            factor (in px) to add to object (default: 0)
        thresholding: {'binary', 'adaptive', 'otsu'}
            type of thresholding (default: binary)
        sensitivity: int (odd)
            needs to be specified if 'adaptive' thresholding is used (default: 99)
        iterations: int
            needs to be specified if 'adaptive' thresholding is used (default: 3)
        thresh_val: int (0-255)
            needs to be specified if 'binary' thresholding is used (default: 127)
        min_size: int
            minimum size (longest distance in contour) in pixels for objects to be included (default: 0)
        min_area: int
            minimum contour area in pixels for objects to be included (default: 0)
            

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
        # initialize -----------------
        self.image = cv2.imread(im_path)
        self.filename = os.path.basename(im_path)
        try:
            self.date = exif_date(im_path)
        except:
            self.date = "NA"

        # =============================================================================
        # Thresholding, masking & find contours
        # =============================================================================
        blur_kern = kwargs.get('blur', 99)
        thresholding = kwargs.get('thresholding', "otsu")
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)       
        self.blurred = blur(self.gray, blur_kern)

        # thresholding - apply correction factor
        if thresholding == "otsu":
            ret, self.thresh = cv2.threshold(self.blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif thresholding == "adaptive":
            sensitivity = kwargs.get('sensitivity', 99)
            iterations = kwargs.get('iterations', 3)
            self.thresh = cv2.adaptiveThreshold(self.blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,sensitivity, iterations)
        elif thresholding == "binary":
            value = kwargs.get('bin_thresh', 127)
            ret, self.thresh = cv2.threshold(self.blurred,value, 255,cv2.THRESH_BINARY_INV)
        corr_factor = kwargs.get('corr_factor', 0)
        if corr_factor < 0:
            corr_factor = abs(corr_factor)
            self.thresh = cv2.erode(self.thresh,np.ones((corr_factor,corr_factor),np.uint8), iterations = 1)
        if corr_factor > 0:
            self.thresh = cv2.dilate(self.thresh,np.ones((corr_factor,corr_factor),np.uint8), iterations = 1)

        # mask arena, scale, etc.
        if "exclude" in kwargs:
            self.mask = sum(kwargs.get('exclude'))
            self.thresh = cv2.subtract(self.thresh,self.mask)
            self.thresh[self.thresh==1] = 0

        # find contours of objects
        idx = 0
        df_list = []
        ret, contours, hierarchy = cv2.findContours(self.thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
        for cnt in contours:
            if len(cnt) > 5:
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                length = int(radius * 2)
                area = int(cv2.contourArea(cnt))
                if length > kwargs.get('min_size', 0) and area > kwargs.get('min_area', 0):
                    idx += 1
                    df_list.append([self.filename, self.date, idx, int(x), int(y), scale, length, area, cnt])

        # =============================================================================
        # Build dataframe 
        # =============================================================================
        self.df = pd.DataFrame(data=df_list, columns = ["file","date","object", "x", "y", "scale","length", "area","contour"])
        self.df.index += 1
        if 'gray_value_ref' in kwargs:
            ret = gray_scale(source=self.gray, mode="image", resize=0.1, ret = True, printing=False)
            ref = kwargs.get('gray_value_ref', 200)
            self.df["gray_corr_factor"] = int(ref - ret)

        mean_list = []
        std_list = []
        bgr_list = []
        bgr_std_list = []

        self.drawn = copy.deepcopy(self.image)

        for idx, row in self.df.iterrows():
            # roi masking + extraction
            rx,ry,rw,rh = cv2.boundingRect(row["contour"])
            grayscale =  ma.array(data=self.gray[ry:ry+rh,rx:rx+rw], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
            b =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,0], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
            g =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,1], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
            r =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,2], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
            mean_list.append(int(np.mean(grayscale))) # mean grayscale value
            std_list.append(int(np.std(grayscale))) # standard deviation of grayscale values
            bgr_list.append((int(np.mean(b)),int(np.mean(g)),int(np.mean(r)))) # mean grayscale value
            bgr_std_list.append((int(np.std(b)),int(np.std(g)),int(np.std(r)))) # mean grayscale value

            # drawing
            q=kwargs.get("roi_size",300)/2
            x, y = (row["x"], row["y"])
            cv2.rectangle(self.drawn,(int(max(0,x-q)),int(max(0, y-q))),(int(min(self.image.shape[1],x+q)),int(min(self.image.shape[0],y+q))),red,8)
            cv2.putText(self.drawn,  str(int(row["object"])) ,(int(row["x"]),int(row["y"])), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),5,cv2.LINE_AA)
            cv2.drawContours(self.drawn, [row["contour"]], 0, green, 4)

        self.df["mean1"] = mean_list
        self.df["sd1"] = std_list
        self.df["bgr1"] = bgr_list
        self.df["bgr_sd1"] = bgr_std_list
        
        # =============================================================================
        # finish and return df and image
        # =============================================================================
        if hasattr(self,'mask'):
            boo = np.array(self.mask, dtype=bool)
            #self.drawn = copy.deepcopy(self.image)
            self.drawn[boo,2] = 255
        if kwargs.get('show', False) == True:
            cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
            cv2.imshow('phenopype', self.drawn)
            cv2.waitKey(200)

            
        # return things
        self.df = self.df.loc[:, self.df.columns != 'contour']
        self.df_short = self.df[["x", "y","scale", "length", "area","mean1","bgr1"]]
        print("----------------------------------------------------------")
        print("Found " + str(len(self.df)) + " objects in " + self.filename)
        print("----------------------------------------------------------")
        print(self.df_short) # 



    def save(self, **kwargs):
        # set dir and names
        out_dir = kwargs.get("out_dir",os.path.join(os.getcwd(),"out")) # "out") #
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
        if "image" in kwargs:
            if kwargs.get('overwrite',True) == False:
                if not os.path.exists(im_path):
                    cv2.imwrite(im_path, kwargs.get("image"))
            else:
                cv2.imwrite(im_path, kwargs.get("image"))

        # df
        if "df" in kwargs:
            if kwargs.get('overwrite',True) == False:
                if not os.path.exists(df_path):
                    kwargs.get("df").to_csv(path_or_buf=df_path, sep="\t", index=False)
            else:
                    kwargs.get("df").to_csv(path_or_buf=df_path, sep="\t", index=False)





