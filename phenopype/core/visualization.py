#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

import math

from phenopype.settings import *
from phenopype.utils_lowlevel import _auto_line_width, _auto_point_size, _auto_text_width, _auto_text_size

#%% settings

inf = math.inf

#%% functions

def select_canvas(obj_input, **kwargs):
    
    ## kwargs
    canvas = kwargs.get("canvas", "image_mod")

    ## method
    if canvas == "bin" or canvas == "binary":
        obj_input.canvas = copy.deepcopy(obj_input.image_bin)
        print("- binary image")
    elif canvas == "gray" or canvas == "image_gray":
        if obj_input.image_gray.__class__.__name__ == "NoneType":
            obj_input.image_gray = cv2.cvtColor(obj_input.image_copy,cv2.COLOR_BGR2GRAY)
        obj_input.canvas = copy.deepcopy(obj_input.image_gray)
        print("- grayscale image")
    elif canvas in ["image_mod", "mod"]:
        obj_input.canvas = copy.deepcopy(obj_input.image)
        print("- modifed image")
    elif canvas == "image_raw":
        obj_input.canvas = copy.deepcopy(obj_input.image_copy)
        print("- raw image")
    elif canvas == "g" or canvas == "green":
        obj_input.canvas = copy.deepcopy(obj_input.image_copy[:,:,0])
        print("- green channel")
    elif canvas == "r" or canvas == "red":
        obj_input.canvas = copy.deepcopy(obj_input.image_copy[:,:,1])
        print("- red channel")
    elif canvas == "b" or canvas == "blue":
        obj_input.canvas = copy.deepcopy(obj_input.image_copy[:,:,2])
        print("- blue channel")
        
    ## check if 3D
    if len(obj_input.canvas.shape)<3:
        obj_input.canvas = cv2.cvtColor(obj_input.canvas, cv2.COLOR_GRAY2BGR)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return obj_input



def show_contours(obj_input,**kwargs):

    ## kwargs
    df_image_data = kwargs.get("df", None)
    df_contours = kwargs.get("df_contours", None)
    offset_coords = kwargs.get("offset_coords", None)
    flag_label = kwargs.get("label", True)
    flag_fill = kwargs.get("fill", 0.2)
    flag_child = kwargs.get("mark_holes", True)
    level = kwargs.get("level", 3)
    line_colour_sel = colours[kwargs.get("line_colour", "green")]
    text_colour = colours[kwargs.get("text_colour", "black")]

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"}, index=[0])
        if df_contours.__class__.__name__ == "NoneType":
            warnings.warn("No contour df provided - cannot draw contours.")
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_contours = obj_input.df_contours
    else:
        warnings.warn("wrong input format.")
        return

    ## more kwargs
    flag_line_thickness = kwargs.get("line_thickness", _auto_line_width(image))
    text_thickness = kwargs.get("text_thickness", _auto_line_width(image))
    text_size = kwargs.get("text_size", _auto_text_size(image))

    ## method
    idx = 0
    colour_mask = copy.deepcopy(image)
    for index, row in df_contours.iterrows():
        if flag_child:
            if row["order"] == "child":
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
                    contours=[row["coords"]], 
                    contourIdx = idx,
                    thickness=-1, 
                    color=fill_colour, 
                    maxLevel=level,
                    offset=offset_coords)
        if flag_line_thickness > 0: 
            cv2.drawContours(image=image, 
                    contours=[row["coords"]], 
                    contourIdx = idx,
                    thickness=flag_line_thickness, 
                    color=line_colour, 
                    maxLevel=level,
                    offset=offset_coords)
        if flag_label:
            cv2.putText(image, row["contour"] , (row["center"]), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_size, text_colour, text_thickness, cv2.LINE_AA)
            cv2.putText(colour_mask, row["contour"] , (row["center"]), cv2.FONT_HERSHEY_SIMPLEX, 
                        text_size, text_colour, text_thickness, cv2.LINE_AA)
    image = cv2.addWeighted(image,1-flag_fill, colour_mask, flag_fill, 0) 

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = image




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
        df_landmarks = obj_input.df_landmarks

    point_size = kwargs.get("point_size", _auto_point_size(image))
    point_col = kwargs.get("point_col", "red")
    label_size = kwargs.get("label_size", _auto_text_size(image))
    label_width = kwargs.get("label_width", _auto_text_width(image))
    label_col = kwargs.get("label_col", "black")

    ## visualize
    for label, x, y in zip(df_landmarks.landmark, df_landmarks.x, df_landmarks.y):
        cv2.circle(image, (x,y), point_size, colours[point_col], -1)
        cv2.putText(image, str(label), (x,y), 
            cv2.FONT_HERSHEY_SIMPLEX, label_size, colours[label_col], label_width, cv2.LINE_AA)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        if  obj_input.canvas.__class__.__name__ == "ndarray":
            obj_input.canvas = image
        else:
            obj_input.image = image



def show_masks(obj_input, **kwargs):
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
    include = kwargs.get("include", None)

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_masks = obj_input.df_masks

    ## more kwargs
    line_width = kwargs.get("line_thickness", _auto_line_width(image))

    ## draw masks from mask obect
    for index, row in df_masks.iterrows():
            if not include.__class__.__name__ == "NoneType":
                if row["mask"] in include:
                    pass
                else:
                    continue
            else:
                pass
            print(" - show mask: " + row["mask"] + ".")
            coords = eval(row["coords"])
            if not row["mask"] == "scale":
                col = colours["blue"]
            else:
                col = colours["red"]
            cv2.polylines(image, np.array([coords]), False, col, line_width)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        if  obj_input.canvas.__class__.__name__ == "ndarray":
            obj_input.canvas = image
        else:
            obj_input.image = image



def show_polylines(obj_input, **kwargs):
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
    colour = kwargs.get("colour", "green")

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_polylines = obj_input.df_polylines
        
    ## more kwargs
    line_width = kwargs.get("line_thickness", _auto_line_width(image))

    ## visualize
    for polyline in df_polylines["polyline"].unique():
        sub = df_polylines.groupby(["polyline"])
        sub = sub.get_group(polyline)
        coords = list(sub[["x","y"]].itertuples(index=False, name=None))
        cv2.polylines(image, np.array([coords]), 
                      False, colours[colour], line_width)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        if  obj_input.canvas.__class__.__name__ == "ndarray":
            obj_input.canvas = image
        else:
            obj_input.image = image