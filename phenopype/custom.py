# -*- coding: utf-8 -*-
"""
Created: 2016/03/31
Last Update: 2018/10/02
Version 0.4.4
@author: Moritz Lürig
"""

import copy
import numpy as np
import numpy.ma as ma
import pandas as pd

import cv2

from phenopype.utils import (blue, green, white)
from phenopype.utils import (blur)

#%%
class kims_module:
    def run(self,image, df, **kwargs):
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
        Returns (optional)
        ------------------
        image: array_like
            original input image (for further referencing)
    
        """
        
        # =============================================================================
        # INITIALIZE
        # =============================================================================
        
        # LOAD
        self.df_list = []
        if "compare_with" in kwargs:
            self.drawn = copy.deepcopy(kwargs.get("compare_with"))
        else:
            self.drawn = copy.deepcopy(image)
            
        # LOOP THROUGH PROVIDED DF
        for idx, row in df.iterrows():
            roi_size = kwargs.get("roi_size", 500)
            q=roi_size/2
            x = int(row["x"])
            y = int(row["y"])
            self.roi = image[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))]
            self.roi_gray = cv2.cvtColor(self.roi,cv2.COLOR_BGR2GRAY)
            self.roi_drawn = self.drawn[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))]
    
             
            # =============================================================================
            # BLUR1 > THRESHOLDING > MORPHOLOGY > BLUR2
            # =============================================================================
                
            # BLUR1
            if "blur1" in kwargs:
                blur_kernel = kwargs.get("blur1")
                self.roi_blurred = blur(self.roi_gray, blur_kernel)
            else: 
                self.roi_blurred = self.roi_gray   
            
            # THRESHOLDING
            thresholding = kwargs.get('thresholding', "otsu")
            if thresholding == "otsu":
                ret, self.thresh = cv2.threshold(self.roi_blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif thresholding == "adaptive":
                sensitivity = kwargs.get('sensitivity', 33)
                iterations = kwargs.get('iterations', 3)
                self.thresh = cv2.adaptiveThreshold(self.roi_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,sensitivity, iterations)     
            elif thresholding == "binary":
                value = kwargs.get('bin_thresh', 127)
                ret, self.thresh = cv2.threshold(self.roi_blurred,value, 255,cv2.THRESH_BINARY_INV)
            self.morph = self.thresh
    
#            # MORPHOLOGY
            if "erosion" in kwargs:
                err_factor = kwargs.get('erosion')
                err_element = 5; err_iter = int((row["length"] * 0.01)  *  err_factor)
                err_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(err_element,err_element))
                self.morph = cv2.morphologyEx(self.morph,cv2.MORPH_OPEN,err_kernel, iterations = err_iter)

            if "dilation" in kwargs:
                dil_factor = kwargs.get('dilation')
                dil_element = 5; dil_iter = int((row["length"] * 0.01)  * dil_factor)
                dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dil_element,dil_element))
                self.morph = cv2.morphologyEx(self.morph,cv2.MORPH_CLOSE,dil_kernel, iterations = dil_iter)
                
            # BLUR2
            if "blur2" in kwargs:
                blur_kernel, thresh_val = kwargs.get("blur2")
                self.morph = blur(self.morph, blur_kernel)
                ret, self.morph = cv2.threshold(self.morph, thresh_val, 255,cv2.THRESH_BINARY)

                
                
            # get contours
            ret1, contours1, hierarchy1 = cv2.findContours(copy.deepcopy(self.morph),cv2.RETR_LIST ,cv2.CHAIN_APPROX_TC89_L1)       
            if contours1:
                areas = [cv2.contourArea(cnt1) for cnt1 in contours1]                
                shape = contours1[np.argmax(areas)]   
                (cx,cy),radius = cv2.minEnclosingCircle(shape)
                length2 = int(radius * 2) 
                area2 = int(cv2.contourArea(shape))
    
                # extract info from masked image
                grayscale =  ma.array(data=self.roi_gray, mask = np.logical_not(self.morph))
                b =  ma.array(data=self.roi[:,:,0], mask = np.logical_not(self.morph))
                g =  ma.array(data=self.roi[:,:,1], mask = np.logical_not(self.morph))
                r =  ma.array(data=self.roi[:,:,2], mask = np.logical_not(self.morph))
                
                if len(np.unique(self.morph, return_counts=True)[1]) > 1:
                    mean2 = int(np.mean(grayscale)) # mean grayscale value
                    sd2 = int(np.std(grayscale)) # standard deviation of grayscale values
                    bgr2 = (int(np.mean(b)),int(np.mean(g)),int(np.mean(r))) # mean grayscale value
                    bgr_sd2 = (int(np.std(b)),int(np.std(g)),int(np.std(r))) # mean grayscale value
                else:
                    mean2 = "NA"
                    sd2 = "NA"
                    bgr2 = "NA"
                    bgr_sd2 = "NA"
                self.df_list.append([length2, area2, mean2, sd2, bgr2, bgr_sd2])

                cv2.drawContours(self.roi_drawn, [shape], 0, green, 4)
                if kwargs.get("label", False) == True:
                    cv2.putText(self.roi_drawn, str(int(row["idx"])) ,(int(cx),int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 4,white,5,cv2.LINE_AA)
                self.drawn[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))] = self.roi_drawn  

        self.df = pd.DataFrame(data=self.df_list, columns = ["length2", "area2", "mean2", "sd2", "bgr2", "bgr_sd2"])
      
        if kwargs.get('show', False) == True:
            cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
            cv2.imshow('phenopype', self.drawn)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


#%%

class camera_stand_module:
    def run(self, image, df, **kwargs):
        
        if "compare_to" in kwargs:
            self.drawn = copy.deepcopy(kwargs.get("compare_to"))
        else:
            self.drawn = copy.deepcopy(image)

        # =============================================================================
        # make ROIs of input images
        # =============================================================================
        for idx, row in df.iterrows():
            roi_size = kwargs.get("roi_size", 500)
            q=roi_size/2
            x = int(row["x"])
            y = int(row["y"])
            self.roi = image[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))]
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            self.roi_gray = gray[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))]
            self.roi_drawn = self.drawn[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))]
    
            if "blur1" in kwargs:
                blur_kernel = kwargs.get("blur1")
                self.roi_blurred = blur(self.roi_gray, blur_kernel)
            else: 
                self.roi_blurred = self.roi_gray
            # thresholding
            thresholding = kwargs.get('thresholding', "otsu")
            if thresholding == "otsu":
                ret, self.thresh = cv2.threshold(self.roi_blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif thresholding == "adaptive":
                sensitivity = kwargs.get('sensitivity', 33)
                iterations = kwargs.get('iterations', 3)
                self.thresh = cv2.adaptiveThreshold(self.roi_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,sensitivity, iterations)     
            elif thresholding == "binary":
                value = kwargs.get('bin_thresh', 127)
                ret, self.thresh = cv2.threshold(self.roi_blurred,value, 255,cv2.THRESH_BINARY_INV)

#            # morphological operations
#            if "erosion" in kwargs:
#                err_factor = kwargs.get('erosion')
#                err_element = 5; err_iter = int((row["length"] * 0.01)  *  err_factor)
#                err_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(err_element,err_element))
#                self.morph = cv2.morphologyEx(self.thresh,cv2.MORPH_OPEN,err_kernel, iterations = err_iter)
#
#            elif "dilation" in kwargs:
#                dil_factor = kwargs.get('dilation')
#                dil_element = 5; dil_iter = int((row["length"] * 0.01)  * dil_factor)
#                dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dil_element,dil_element))
#                self.morph = cv2.morphologyEx(self.thresh,cv2.MORPH_CLOSE,dil_kernel, iterations = dil_iter)

            if "blur2" in kwargs:
                blur_kernel, thresh_val = kwargs.get("blur2")
                self.thresh = blur(self.thresh, blur_kernel)
                ret, self.morph = cv2.threshold(self.thresh, thresh_val, 255,cv2.THRESH_BINARY)
            else:
                self.morph = self.thresh
                
            # get contours
            ret, contours, hierarchy = cv2.findContours(copy.deepcopy(self.morph),cv2.RETR_LIST ,cv2.CHAIN_APPROX_TC89_L1)       
            if contours:
                areas = [cv2.contourArea(cnt) for cnt in contours]                
                cnt = contours[np.argmax(areas)]
                (cx,cy),radius = cv2.minEnclosingCircle(cnt)
                length2 = int(radius * 2) 
                area2 = int(cv2.contourArea(cnt))
    
                # extract info from masked image
                grayscale =  ma.array(data=self.roi_gray, mask = np.logical_not(self.morph))
                b =  ma.array(data=self.roi[:,:,0], mask = np.logical_not(self.morph))
                g =  ma.array(data=self.roi[:,:,1], mask = np.logical_not(self.morph))
                r =  ma.array(data=self.roi[:,:,2], mask = np.logical_not(self.morph))
                
                if len(np.unique(self.morph, return_counts=True)[1]) > 1:
                    mean2 = int(np.mean(grayscale)) # mean grayscale value
                    sd2 = int(np.std(grayscale)) # standard deviation of grayscale values
                    bgr2 = (int(np.mean(b)),int(np.mean(g)),int(np.mean(r))) # mean grayscale value
                    bgr_sd2 = (int(np.std(b)),int(np.std(g)),int(np.std(r))) # mean grayscale value
                else:
                    mean2 = "NA"
                    sd2 = "NA"
                    bgr2 = "NA"
                    bgr_sd2 = "NA"
                self.df_list.append([row["filename","date_taken", "date_analyzed", "idx", "x", "y", "scale"] + [length2, area2, mean2, sd2, bgr2, bgr_sd2]])


                if kwargs.get("label", False) == True:
                    cv2.putText(self.roi_drawn, str(int(row["idx"])) ,(int(row["x"]),int(row["y"])), cv2.FONT_HERSHEY_SIMPLEX, 4,white,5,cv2.LINE_AA)
                if "compare_to" in kwargs:
                    cv2.drawContours(self.roi_drawn, [cnt], 0, green, 4)
                    self.drawn[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))] = self.roi_drawn  
                else:
                    if "gray_corr_factor" in list(df):
                        self.roi_drawn = self.roi_gray + row["gray_corr_factor"]
                        self.roi_drawn = np.array(self.roi_drawn, dtype="uint8")
                    else: 
                        self.roi_drawn = self.roi_gray
                    self.roi_drawn = cv2.cvtColor(self.roi_drawn,cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(self.roi_drawn, [cnt], 0, green, 4)
                    
            
                    
            self.df = pd.DataFrame(data=self.df_list, columns = ["filename","date_taken", "date_analyzed", "idx", "x", "y", "scale","length2", "area2", "mean2", "sd2", "bgr2", "bgr_sd2"])
            self.df = pd.DataFrame(data=self.df_list, columns = ["length2", "area2", "mean2", "sd2", "bgr2", "bgr_sd2"])

                
            if kwargs.get('show', False) == True:
                cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
                if "compare_to" in kwargs:
                    cv2.imshow('phenopype', self.drawn)
                else:
                    cv2.imshow('phenopype', self.roi_drawn)

            else:
                df.reindex(columns = ["length2", "area2", "mean2", "sd2", "bgr2", "bgr_sd2"])


# =============================================================================
# 
# =============================================================================
    
#roi = img[max(0,y-q):y+q,max(0,x-q):x+q]   # img[y-400:y+400, x-400:x+400] 
#                else:
#                    roi = img
#                    
#                # ii) actual phenotypinmg
#                morph2 = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,phenotyping_value,phenotyping_iterations)
#                morph2 = cv2.morphologyEx(morph2,cv2.MORPH_CLOSE,np.ones((kernel_close),np.uint8), iterations = iterations_close)
#                morph2 = cv2.morphologyEx(morph2,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,kernel_open), iterations = iterations_open)
#                ret2, contours2, hierarchy2 = cv2.findContours(copy.deepcopy(morph2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)    
#            
#                # continue ONLY make files IF countour exists, otherwise don't add line to text file BUT make image
#                if contours2: 
#                    areas2 = [cv2.contourArea(cnt) for cnt in contours2]  # list of contours in ROI
#                    largest2 = contours2[np.argmax(areas2)]               # largest contour in ROI
#                
#                # combine contours from multiple detection procedures
#                # conc = np.concatenate((largest1, largest2), axis=0)
#                    conc = largest2

