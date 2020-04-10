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

def select_canvas(obj_input, canvas="image_mod", multi=True):
    """
    Isolate a colour channel from an image or select canvas for the pype method.

    Parameters
    ----------
    obj_input : array or container
        input object
    canvas : {"mod", "bin", "gray", "raw", "red", "green", "blue"} str, optional
        the type of canvas to be used for visual feedback. some types require a
        function to be run first, e.g. "bin" needs a segmentation algorithm to be
        run first. black/white images don't have colour channels. coerced to 3D
        array by default
    multi: bool, optional
        coerce returned array to multilevel (3-level)

    Returns
    -------
    obj_input : container
        canvas can be called with "obj_input.canvas".

    """
    ## kwargs
    flag_multi = multi

    ## load image
    flag_container = False
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        flag_container = True
        image = copy.deepcopy(obj_input.image_copy)
    else:
        print("wrong input format.")
        return

    ## method
    ## for container
    if flag_container and canvas in ["image_bin", "bin", "binary"]:
        canvas = copy.deepcopy(obj_input.image_bin)
        print("- binary image")
    elif flag_container and canvas in ["image_mod", "mod", "modified"]:
        canvas = copy.deepcopy(obj_input.image)
        print("- modifed image")
    elif flag_container and canvas in ["image_raw", "raw"]:
        canvas = copy.deepcopy(obj_input.image_copy)
        print("- raw image")
    ## for array and container
    elif canvas in ["gray", "grey", "image_gray"]:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("- grayscale image")
    elif canvas in ["g","green"]:
        canvas = image[:,:,0]
        print("- green channel")
    elif canvas in ["r","red"]:
        canvas = image[:,:,1]
        print("- red channel")
    elif canvas in ["b","blue"]:
        canvas = image[:,:,2]
        print("- blue channel")
    else:
        print("- invalid selection - defaulting to raw image")
        canvas = copy.deepcopy(obj_input.image_copy)

    ## check if 3D
    if flag_multi:
        if len(canvas.shape)<3:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return canvas
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = canvas


def show_contours(obj_input, df_image_data=None, df_contours=None, offset_coords=None,
                  label=True, fill=0.3, mark_holes=True, level=3, line_colour="green",
                  label_size=None, label_colour="black", line_width=None, label_width=None):
    """
    

    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    df_image_data : TYPE, optional
        DESCRIPTION. The default is None.
    df_contours : TYPE, optional
        DESCRIPTION. The default is None.
    offset_coords : TYPE, optional
        DESCRIPTION. The default is None.
    label : TYPE, optional
        DESCRIPTION. The default is True.
    fill : TYPE, optional
        DESCRIPTION. The default is 0.3.
    mark_holes : TYPE, optional
        DESCRIPTION. The default is True.
    level : TYPE, optional
        DESCRIPTION. The default is 3.
    line_colour : TYPE, optional
        DESCRIPTION. The default is "green".
    line_width: int, optional
        line width
    label_colour : {"black", "white", "green", "red", "blue"} str, optional
        contour label colour.
    label_size: int, optional
        contour label font size (scaled to image)
    label_width: int, optional
        contour label font thickness 
        
    Returns
    -------
    image: array or container
        image with contours

    """
    ## kwargs
    flag_label = label
    flag_fill = fill
    flag_child = mark_holes
    flag_line_width = line_width
    line_colour_sel = colours[line_colour]
    label_colour = colours[label_colour]

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
        if df_contours.__class__.__name__ == "NoneType":
            print("No df provided - cannot draw contours.")
            return
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_contours = obj_input.df_contours
    else:
        print("wrong input format.")
        return


    ## more kwargs
    if line_width.__class__.__name__ == "NoneType":
        flag_line_width = _auto_line_width(image)
    if label_width.__class__.__name__ == "NoneType":
        label_width = _auto_text_width(image)
    if label_size.__class__.__name__ == "NoneType":
        label_size = _auto_text_size(image)

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
        if flag_line_width > 0: 
            cv2.drawContours(image=image, 
                    contours=[row["coords"]], 
                    contourIdx = idx,
                    thickness=flag_line_width, 
                    color=line_colour, 
                    maxLevel=level,
                    offset=offset_coords)
        if flag_label:
            cv2.putText(image, row["contour"] , (row["center"]), 
                        cv2.FONT_HERSHEY_SIMPLEX, label_size, label_colour, 
                        label_width, cv2.LINE_AA)
            cv2.putText(colour_mask, row["contour"] , (row["center"]), 
                        cv2.FONT_HERSHEY_SIMPLEX, label_size, label_colour, 
                        label_width, cv2.LINE_AA)

    image = cv2.addWeighted(image,1-flag_fill, colour_mask, flag_fill, 0) 

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = image




def show_landmarks(obj_input, df_landmarks=None, point_col="green", 
                   point_size=None, label_col="black", label_size=None, 
                   label_width=None):
    """
    Draw landmarks into an image.

    Parameters
    ----------
    obj_input : array or container
        input object
    df_landmarks: DataFrame, optional
        should contain contour coordinates as an array in a df cell
    point_col: {"green", "blue", "red", "black", "white"} str, optional
        landmark point colour
    point_size: int, optional
        landmark point size in pixels
    label_col : {"black", "white", "green", "red", "blue"} str, optional
        landmark label colour.
    label_size: int, optional
        landmark label size (scaled to image)
    label_width: int, optional
        text thickness 

    Returns
    -------
    image: array or container
        image with landmarks

    """

    ## kwargs
    point_col = colours[point_col]
    label_col = colours[label_col]

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_landmarks.__class__.__name__ == "NoneType":
            print("No df provided - cannot draw landmarks.")
            return
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_landmarks = obj_input.df_landmarks
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if point_size.__class__.__name__ == "NoneType":
        point_size = _auto_point_size(image)
    if label_size.__class__.__name__ == "NoneType":
        label_size = _auto_text_size(image)
    if label_width.__class__.__name__ == "NoneType":
        label_width = _auto_text_width(image)

    ## visualize
    for label, x, y in zip(df_landmarks.landmark, df_landmarks.x, df_landmarks.y):
        cv2.circle(image, (x,y), point_size, point_col, -1)
        cv2.putText(image, str(label), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 
                    label_size, label_col, label_width, cv2.LINE_AA)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = image



def show_masks(obj_input, select=None, df_masks=None, line_colour="blue", 
               line_width=None, label=False, label_size=None, 
               label_col="black", label_width=None):
    
    
    """Draw masks into an image.
    
    Parameters
    ----------        
    
    select: str or list
        select a subset of masks to display
    df_masks: DataFrame
        contains mask coordinates and label
    label: str (default: "area1")
        passes a label to the mask
    tool: str (default: "rectangle")
        zoom into the scale with "rectangle" or "polygon".
        
    """
    ## kwargs
    flag_label = label
    line_colour_sel = colours[line_colour]
    label_colour = colours[label_col]
    if not select.__class__.__name__ == "NoneType":
        if not select.__class__.__name__ == "list":
            select = [select]

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
        if df_masks.__class__.__name__ == "NoneType":
            print("No df provided - cannot draw masks.")
            return
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_masks = obj_input.df_masks
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if line_width.__class__.__name__ == "NoneType":
        line_width = _auto_line_width(image)
    if label_width.__class__.__name__ == "NoneType":
        label_width = _auto_text_width(image)
    if label_size.__class__.__name__ == "NoneType":
        label_size = _auto_text_size(image)

    ## draw masks from mask obect
    for index, row in df_masks.iterrows():
            if not select.__class__.__name__ == "NoneType":
                if row["mask"] in select:
                    pass
                else:
                    continue
            else:
                pass
            print(" - show mask: " + row["mask"] + ".")
            coords = eval(row["coords"])
            if row["mask"] == "scale":
                line_colour = colours["red"]
            else:
                line_colour = line_colour_sel
            cv2.polylines(image, np.array([coords]), False, line_colour, line_width)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = image




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
        obj_input.canvas = image
