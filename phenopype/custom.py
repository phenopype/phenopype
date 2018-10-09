# -*- coding: utf-8 -*-
"""
Created: 2016/03/31
Last Update: 2018/10/02
Version 0.4.0
@author: Moritz Lürig
"""

import numpy as np
import numpy.ma as ma
import copy

import cv2

from phenopype.utils import (blue, white)
from phenopype.utils import (blur)

#%%

def kims_module(image, df, **kwargs):

    # initialize
    mean_list = []
    std_list = []
    bgr_list = []
    bgr_std_list = []

    if "compare_with" in kwargs:
        drawn = copy.deepcopy(kwargs.get("compare_with"))
    else:
        drawn = copy.deepcopy(image)
        
    # loop through contours in df and make ROIs
    for idx, row in df.iterrows():
        q=150
        x = int(row["x"])
        y = int(row["y"])
        roi = image[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))]
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        roi_gray = gray[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))]
        roi_drawn = drawn[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))]

        if "blur" in kwargs:
            blur_kern = kwargs.get("blur")
            roi_gray = blur(roi_gray, blur_kern)

        # thresholding
        thresholding = kwargs.get('thresholding', "otsu")
        if thresholding == "otsu":
            ret, thresh = cv2.threshold(roi_gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif thresholding == "adaptive":
            sensitivity = kwargs.get('sensitivity', 33)
            iterations = kwargs.get('iterations', 3)
            thresh = cv2.adaptiveThreshold(roi_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,sensitivity, iterations)     
        elif thresholding == "binary":
            value = kwargs.get('bin_thresh', 127)
            ret, thresh = cv2.threshold(roi_gray,value, 255,cv2.THRESH_BINARY_INV)

        # morphological operations
        morph = thresh
        if "erosion" in kwargs:
            err_factor = kwargs.get('erosion')
            err_element = 5; err_iter = int((row["length"] * 0.01)  *  err_factor)
            err_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(err_element,err_element))
            morph = cv2.morphologyEx(morph,cv2.MORPH_OPEN,err_kernel, iterations = err_iter)
            
        if "dilation" in kwargs:
            dil_factor = kwargs.get('dilation')
            dil_element = 5; dil_iter = int((row["length"] * 0.01)  * dil_factor)
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dil_element,dil_element))
            morph = cv2.morphologyEx(morph,cv2.MORPH_CLOSE,dil_kernel, iterations = dil_iter)

        # get contours
        ret1, contours1, hierarchy1 = cv2.findContours(copy.deepcopy(morph),cv2.RETR_LIST ,cv2.CHAIN_APPROX_TC89_L1)       
        if contours1:
            areas = [cv2.contourArea(cnt1) for cnt1 in contours1]                
            shape = contours1[np.argmax(areas)]

        # extract info from masked image
        grayscale =  ma.array(data=roi_gray, mask = np.logical_not(morph))
        b =  ma.array(data=roi[:,:,0], mask = np.logical_not(morph))
        g =  ma.array(data=roi[:,:,1], mask = np.logical_not(morph))
        r =  ma.array(data=roi[:,:,2], mask = np.logical_not(morph))
        
        if len(np.unique(morph, return_counts=True)[1]) > 1:
            mean_list.append(int(np.mean(grayscale))) # mean grayscale value
            std_list.append(int(np.std(grayscale))) # standard deviation of grayscale values
            bgr_list.append((int(np.mean(b)),int(np.mean(g)),int(np.mean(r)))) # mean grayscale value
            bgr_std_list.append((int(np.std(b)),int(np.std(g)),int(np.std(r)))) # mean grayscale value

            cv2.drawContours(roi_drawn, [shape], 0, blue, 4)
            cv2.putText(roi_drawn, str(int(row["object"])) ,(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 4,white,5,cv2.LINE_AA)
            drawn[int(max(0,y-q)):int(min(image.shape[0],y+q)),int(max(0,x-q)):int(min(image.shape[1],x+q))] = roi_drawn  
        else:
            mean_list.append("NA") # mean grayscale value
            std_list.append("NA") # standard deviation of grayscale values
            bgr_list.append("NA") # mean grayscale value
            bgr_std_list.append("NA") # mean grayscale value

    df["mean2"] = mean_list
    df["sd2"] = std_list
    df["bgr2"] = bgr_list
    df["bgr_sd2"] = bgr_std_list

    if kwargs.get('show', False) == True:
        cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
        cv2.imshow('phenopype', drawn)
        if cv2.waitKey(0)==27:
            cv2.destroyAllWindows()

    return df, drawn

