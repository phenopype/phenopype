#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

import math

from phenopype.settings import colours
from phenopype.utils_lowlevel import _auto_line_width, _auto_point_size, _auto_text_width, _auto_text_size, _load_masks

#%% settings

inf = math.inf

#%% functions

def select_canvas(obj_input, **kwargs):
    
    ## kwargs
    canvas = kwargs.get("canvas", "mod")
    
    ## method
    if canvas == "bin" or canvas == "binary":
        obj_input.canvas = copy.deepcopy(obj_input.image_bin)
    if canvas == "gray" or canvas == "grayscale":
        if obj_input.image_gray.__class__.__name__ == "NoneType":
            obj_input.image_gray = cv2.cvtColor(obj_input.image_copy,cv2.COLOR_BGR2GRAY)
        obj_input.canvas = copy.deepcopy(obj_input.image_gray)
    if canvas == "mod" or canvas == "modified":
        obj_input.canvas = copy.deepcopy(obj_input.image)
    if canvas == "img" or canvas == "image":
        obj_input.canvas = copy.deepcopy(obj_input.image_copy)
    if canvas == "g" or canvas == "green":
        obj_input.canvas = copy.deepcopy(obj_input.image_copy[:,:,0])
    if canvas == "r" or canvas == "red":
        obj_input.canvas = copy.deepcopy(obj_input.image_copy[:,:,1])
    if canvas == "b" or canvas == "blue":
        obj_input.canvas = copy.deepcopy(obj_input.image_copy[:,:,2])
        
    if len(obj_input.canvas.shape)<3:
        obj_input.canvas = cv2.cvtColor(obj_input.canvas, cv2.COLOR_GRAY2BGR)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return obj_input

def show_contours(obj_input,**kwargs):

    ## kwargs
    contours = kwargs.get("contours", None)
    flag_label = kwargs.get("label", True)
    flag_fill = kwargs.get("fill", 0.2)
    flag_child = kwargs.get("mark_holes", True)
    level = kwargs.get("level", 3)
    line_colour_sel = colours[kwargs.get("line_colour", "green")]
    text_colour = colours[kwargs.get("text_colour", "black")]
    offset_coords = kwargs.get("offset_coords", None)

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if not contours:
            warnings.warn("No contour list provided - cannot draw contours.")
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        contours = obj_input.contours
        
    ## more kwargs
    flag_line_thickness = kwargs.get("line_thickness", _auto_line_width(image))
    text_thickness = kwargs.get("text_thickness", _auto_line_width(image))
    text_size = kwargs.get("text_size", _auto_text_size(image))

    ## method
    idx = 0
    colour_mask = copy.deepcopy(image)
    for label, contour in contours.items():
        if flag_child:
            if contour["order"] == "child":
                fill_colour = colours["red"]
                line_colour = colours["red"]
            else:
                fill_colour = line_colour_sel
                line_colour = line_colour_sel
        else:
            fill_colour = line_colour_sel
            line_colour = line_colour_sel
        if flag_fill > 0:
            cv2.drawContours(image=colour_mask, 
                    contours=[contour["coords"]], 
                    contourIdx = idx,
                    thickness=-1, 
                    color=fill_colour, 
                    maxLevel=level,
                    offset=offset_coords)
        if flag_line_thickness > 0: 
            cv2.drawContours(image=image, 
                    contours=[contour["coords"]], 
                    contourIdx = idx,
                    thickness=flag_line_thickness, 
                    color=line_colour, 
                    maxLevel=level,
                    offset=offset_coords)
        if flag_label:
            cv2.putText(image, label , (contour["x"],contour["y"]), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_size, text_colour, text_thickness, cv2.LINE_AA)
            cv2.putText(colour_mask, label , (contour["x"],contour["y"]), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_size, text_colour, text_thickness, cv2.LINE_AA)
    image = cv2.addWeighted(image,1-flag_fill, colour_mask, flag_fill, 0) # combine

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        if  obj_input.canvas.__class__.__name__ == "ndarray":
            obj_input.canvas = image
        else:
            obj_input.image = image



def show_landmarks(obj_input, **kwargs):
    """Mask maker method to draw rectangle or polygon mask onto image.
    
    Parameters
    ----------        
    
    include: bool (default: True)
        determine whether resulting mask is to include or exclude objects within
    label: str (default: "area1")
        passes a label to the mask
    tool: str (default: "rectangle")
        zoom into the scale with "rectangle" or "polygon".
        
    """
    

    ## kwargs
    colour = colours[kwargs.get("colour", "green")]
    mask_list = kwargs.get("masks", None)

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas

    point_size = kwargs.get("point_size", _auto_point_size(image))
    point_col = colours[kwargs.get("point_col", "red")]
    text_size = kwargs.get("label_size", _auto_text_size(image))
    text_width = kwargs.get("label_width", _auto_text_width(image))
    text_col = colours[kwargs.get("label_col", "black")]

    ## draw landmarks
    if obj_input.landmarks:
        points = eval(obj_input.landmarks["landmarks"]["coords"])
        for point, idx in zip(points, range(len(points))):
            cv2.circle(image, point, point_size, point_col, -1)
            cv2.putText(image, str(idx+1), point, 
                cv2.FONT_HERSHEY_SIMPLEX, text_size, text_col, text_width, cv2.LINE_AA)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        if  obj_input.canvas.__class__.__name__ == "ndarray":
            obj_input.canvas = image
        else:
            obj_input.image = image



def show_mask(obj_input, **kwargs):
    """Mask maker method to draw rectangle or polygon mask onto image.
    
    Parameters
    ----------        
    
    include: bool (default: True)
        determine whether resulting mask is to include or exclude objects within
    label: str (default: "area1")
        passes a label to the mask
    tool: str (default: "rectangle")
        zoom into the scale with "rectangle" or "polygon".
        
    """
    
    # mask_list = ["mask1"]

    ## kwargs
    colour = colours[kwargs.get("colour", "green")]
    mask_list = kwargs.get("masks", None)

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas

    ## more kwargs
    line_thickness = kwargs.get("line_thickness", _auto_line_width(image))

    ## load masks
    masks, mask_list = _load_masks(obj_input, mask_list)
    if len(masks)==0:
        warnings.warn("No mask-list provided - cannot draw mask outlines.")

    ## draw masks from mask obect
    for mask in masks:
        if mask["label"] in mask_list:
            print(" - applying mask: " + mask["label"] + ".")
            for coord in eval(mask["coords"]):
                image = cv2.polylines(image, [np.array(coord, dtype=np.int32)], False, colour, line_thickness)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        if  obj_input.canvas.__class__.__name__ == "ndarray":
            obj_input.canvas = image
        else:
            obj_input.image = image