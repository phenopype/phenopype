#%% modules
import cv2
import copy
import math
import numpy as np
import os
import pandas as pd
import sys

from phenopype.settings import colours
from phenopype.utils_lowlevel import _load_image, _auto_line_thickness

#%% settings

inf = math.inf

#%% methods



def blur(obj_input, **kwargs):

    ## kwargs
    method = kwargs.get("method","averaging")
    kernel_size = kwargs.get("kernel_size",5)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    sigma_color = kwargs.get("sigma_color",75)
    sigma_space = kwargs.get("sigma_space",75)

    ## load image
    image, flag_input = _load_image(obj_input)

    ## method
    if method=="averaging":
        image_mod = cv2.blur(image,(kernel_size,kernel_size))
    elif method=="gaussian":
        image_mod = cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
    elif method=="median":
        image_mod = cv2.medianBlur(image,kernel_size)
    elif method=="bilateral":
        image_mod = cv2.bilateralFilter(image,kernel_size,sigma_color,sigma_space)
    else:
        image_mod = image

    ## return
    if flag_input=="pype_container":
        obj_input.image_mod = image_mod
    else:
        return image_mod


def find_contours(obj_input, **kwargs):
    
    ## kwargs
    retr = kwargs.get("retrieval", "ext")
    retr_alg = {"ext": cv2.RETR_EXTERNAL, ## only external
                "list": cv2.RETR_LIST, ## all contours
                "tree": cv2.RETR_TREE, ## fully hierarchy
                "ccomp": cv2.RETR_CCOMP, ## outer perimeter and holes
                "flood": cv2.RETR_FLOODFILL} ## not sure what this does
    approx = kwargs.get("approximation", "simple")
    approx_alg = {"none": cv2.CHAIN_APPROX_NONE, ## no approximation of the contours, all points
                "simple": cv2.CHAIN_APPROX_SIMPLE,  ## minimal corners
                "L1": cv2.CHAIN_APPROX_TC89_L1, ## algorithm 1
                "KCOS": cv2.CHAIN_APPROX_TC89_KCOS} ## algorithm 2
    offset_coords = kwargs.get("offset_coords", (0,0))
    min_nodes, max_nodes = kwargs.get('min_nodes', 3), kwargs.get('max_nodes', inf)
    min_diameter, max_diameter = kwargs.get('min_diameter', 0), kwargs.get('max_diameter', inf)
    min_area, max_area = kwargs.get('min_area', 0), kwargs.get('max_area', inf)

    ## load image and check if binary exists
    if obj_input.__class__.__name__ == "pype_container":
        if not obj_input.image_bin.__class__.__name__ == "ndarray":
            threshold(obj_input, colourspace="gray")
        image, flag_input = _load_image(obj_input)

    ## method
    image_mod, contours, hierarchy = cv2.findContours(image=image, 
                                                mode=retr_alg[retr],
                                                method=approx_alg[approx],
                                                offset=offset_coords)
        
    ## filtering
    if contours:
        contour_binder = {}
        contour_df = {}
        idx = 0
        for contour, hier in zip(contours,hierarchy[0]):
            ## number of contour nodes
            if len(contour) > min_nodes and len(contour) < max_nodes:
                center, radius = cv2.minEnclosingCircle(contour)
                x,y = int(center[0]), int(center[1])
                diameter = int(radius*2)
                area = int(cv2.contourArea(contour))
                if all([
                    ## contour diameter
                    diameter > min_diameter and diameter < max_diameter,
                    ## contour area
                    area > min_area and area < max_area
                    ]):
                    
                        idx += 1
                        contour_label = str(idx).zfill(3)
                        contour_df[contour_label] = {"label":contour_label, 
                                                     "x":x,
                                                     "y":y,
                                                     "diameter": diameter, 
                                                     "area":area}
                        hier = [hier[2], hier[3]]
                        contour_binder[contour_label] = {"contour_points":contour,
                                          "contour_hierarchy":hier}
    else:
        contours, hierarchy, contour_df, contour_binder = [], [], {}, {}
        
    contour_df = pd.DataFrame(contour_df).T
    
    ## return
    if flag_input=="pype_container":
        obj_input.contours = contours
        obj_input.hierarchy = hierarchy
        obj_input.contour_df = contour_df
        obj_input.contour_binder = contour_binder




def morphology(obj_input, **kwargs):
    
    ## kwargs   
    kernel_size = kwargs.get("kernel_size", 5)
    shape = kwargs.get("shape", "rect")
    shape_list = {"cross": cv2.MORPH_CROSS, 
                "rect": cv2.MORPH_RECT, 
                "ellipse": cv2.MORPH_ELLIPSE}
    kernel = cv2.getStructuringElement(shape_list[shape], (kernel_size, kernel_size))
    operation = kwargs.get("operation", "close")
    operation_list = {"erode": cv2.MORPH_ERODE, 
                      "dilate": cv2.MORPH_DILATE,
                      "open": cv2.MORPH_OPEN, 
                      "close": cv2.MORPH_CLOSE, 
                      "gradient": cv2.MORPH_GRADIENT,
                      "tophad ": cv2.MORPH_TOPHAT, 
                      "blackhat": cv2.MORPH_BLACKHAT, 
                      "hitmiss": cv2.MORPH_HITMISS}  
    iterations = kwargs.get("iterations", 1)
    
    ## load image
    image, flag_input = _load_image(obj_input)
    
    ## method
    image_mod = cv2.morphologyEx(image, 
                                 op=operation_list[operation], 
                                 kernel = kernel,
                                 iterations = iterations)

    ## return
    if flag_input=="pype_container":
        obj_input.image_mod = image_mod
    else:
        return image_mod



def threshold(obj_input, **kwargs):
    
    ## kwargs
    blocksize = kwargs.get("blocksize", 99)
    constant = kwargs.get("constant", 1)
    colourspace = kwargs.get("colourspace", "gray")
    method = kwargs.get("method", "otsu")
    value = kwargs.get("value", 127)

    ## load image
    image, flag_input = _load_image(obj_input)

    ## colourspace
    if len(image.shape)==3:
        if colourspace == "gray" or colourspace=="grayscale":
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        elif colourspace == "g" or colourspace== "green":
            image = image[:,:,0]
        elif colourspace == "r" or colourspace== "red":
            image = image[:,:,1]
        elif colourspace == "blue" or colourspace== "b":
            image = image[:,:,2]
            
    ## method
    if method == "otsu":
        ret, image_mod = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        image_mod = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blocksize, constant)
    elif method == "binary":
        ret, image_mod = cv2.threshold(image, value, 255,cv2.THRESH_BINARY_INV)  
    else:
        image_mod = image

    ## apply mask
    if flag_input=="pype_container":
        for key, value in obj_input.mask_binder.items():
            MO = value
            if MO.include == True:
                image_mod[np.invert(MO.mask_bool)] = 0

    ## return
    if flag_input=="pype_container":
        obj_input.image_bin = image_mod
        obj_input.image_mod = image_mod
    else:
        return image_mod
