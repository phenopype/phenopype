#%% modules
import cv2
import copy
import math
import numpy as np

from phenopype.settings import colours
from phenopype.utils_lowlevel import _auto_line_thickness, _auto_text_thickness, _auto_text_size, _load_image

#%% settings

inf = math.inf

#%% methods


def show_image(obj_input, **kwargs):
    
    ## kwargs
    canvas = kwargs.get("canvas", "mod")
    
    ## method
    if canvas == "bin" or canvas == "binary":
        obj_input.canvas = copy.deepcopy(obj_input.image_bin)
    if canvas == "gray" or canvas == "grayscale":
        obj_input.canvas = copy.deepcopy(obj_input.image_gray)
    if canvas == "mod" or canvas == "modified":
        obj_input.canvas = copy.deepcopy(obj_input.image_mod)
    if canvas == "img" or canvas == "image":
        obj_input.canvas = copy.deepcopy(obj_input.image)
    if canvas == "g" or canvas == "green":
        obj_input.canvas = copy.deepcopy(obj_input.image[:,:,0])
    if canvas == "r" or canvas == "red":
        obj_input.canvas = copy.deepcopy(obj_input.image[:,:,1])
    if canvas == "b" or canvas == "blue":
        obj_input.canvas = copy.deepcopy(obj_input.image[:,:,2])
        
    if len(obj_input.canvas.shape)<3:
        obj_input.canvas = cv2.cvtColor(obj_input.canvas, cv2.COLOR_GRAY2BGR)


    return obj_input


def show_contours(obj_input, contour_list=[],**kwargs):

    ## kwargs
    offset_coords = kwargs.get("offset_coords", None)
    level = kwargs.get("level", 3)
    line_colour = eval("colours." + kwargs.get("line_colour", "green"))
    text_colour = eval("colours." + kwargs.get("text_colour", "black"))

    ## load image
    image, flag_input = _load_image(obj_input, load="canvas")
    flag_line_thickness = kwargs.get("line_thickness", _auto_line_thickness(image))
    flag_fill = kwargs.get("fill", 0.2)
    text_thickness = kwargs.get("text_thickness", _auto_text_thickness(image))
    text_size = kwargs.get("text_size", _auto_text_size(image))

    if flag_input == "pype_container":
        contour_binder = obj_input.contour_binder
        contour_df = obj_input.contour_df

    # ## method
    # if any(isinstance(i, list) for i in contour_list):
    #     contours = []
    #     for sublist in contour_list:
    #         for item in sublist:
    #             contours.append(item)
    # else:
    #     contours = contour_list
        
    idx = 0
    colour_mask = copy.deepcopy(image)

    for label, contour in contour_binder.items():
        if not contour["contour_hierarchy"][1] == -1:
            fill_colour = colours.red
            line_colour = colours.red
        else:
            fill_colour = line_colour
        if flag_fill > 0:
            cv2.drawContours(image=colour_mask, 
                    contours=[contour["contour_points"]], 
                    contourIdx = idx,
                    thickness=-1, 
                    color=fill_colour, 
                    maxLevel=level,
                    offset=offset_coords)
        if flag_line_thickness > 0: 
            cv2.drawContours(image=image, 
                    contours=[contour["contour_points"]], 
                    contourIdx = idx,
                    thickness=flag_line_thickness, 
                    color=line_colour, 
                    maxLevel=level,
                    offset=offset_coords)
        if label:
            centroid_coords = (contour_df.loc[label]["x"],contour_df.loc[label]["y"])
            cv2.putText(image, label , centroid_coords, cv2.FONT_HERSHEY_SIMPLEX, 
                        text_size, text_colour, text_thickness, cv2.LINE_AA)
            cv2.putText(colour_mask, label , centroid_coords, cv2.FONT_HERSHEY_SIMPLEX, 
                        text_size, text_colour, text_thickness, cv2.LINE_AA)
    image_mod = cv2.addWeighted(image,1-flag_fill, colour_mask, flag_fill, 0) # combine

    ## return
    if flag_input == "pype_container":
        if isinstance(obj_input.canvas, np.ndarray):
            obj_input.canvas = image_mod
        else:
            obj_input.image_mod = image_mod
    else:
        return image_mod
    
    

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

    ## kwargs & load image
    image, flag_input = _load_image(obj_input, load="canvas")
    line_thickness = kwargs.get("line_thickness", _auto_line_thickness(image))
    colour = eval("colours." + kwargs.get("colour", "green"))

    if flag_input == "pype_container":
        mask_binder = obj_input.mask_binder
        mask_filter = kwargs.get("filter", mask_binder)
    else:
        mask_filter = kwargs.get("filter", {})

    ## draw masks from mask obect    
    image_mod = image
    for key, value in mask_binder.items():
        if key in mask_filter:
            MO = value
            for (rx1, ry1, rx2, ry2) in MO.mask_list:
                cv2.rectangle(image, (rx1,ry1), (rx2,ry2), colour, line_thickness)

    ## return
    if flag_input == "pype_container":
        if isinstance(obj_input.canvas, np.ndarray):
            obj_input.canvas = image_mod
        else:
            obj_input.image_mod = image_mod
    else:
        return image_mod