#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours
from phenopype.utils import load_image, load_meta_data, show_image, save_image
from phenopype.utils_lowlevel import _image_viewer, _create_mask_bin
from phenopype.utils_lowlevel import _load_yaml, _show_yaml, _save_yaml, _yaml_file_monitor

#%% methods

def create_mask(obj_input, **kwargs):
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
    skip = False
    label = kwargs.get("label","mask1")
    max_dim = kwargs.get("max_dim", 1000)
    include = kwargs.get("include",True)
    flag_show = kwargs.get("show",False)
    flag_tool = kwargs.get("tool","rectangle")
    flag_overwrite = kwargs.get("overwrite", False)
    
    ## load image and check if pp-project
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image_copy

    ## check if mask exists 
    if obj_input.__class__.__name__ == "container":
        if obj_input.masks:
            masks = dict(obj_input.masks)
            for k, v in masks.items():
                if label == k and not flag_overwrite:
                    include = v["include"]
                    coords = eval(v["coords"])
                    skip = True

    ## method
    if not skip:
        iv_object = _image_viewer(image, 
                                  mode="interactive", 
                                  max_dim = max_dim, 
                                  tool=flag_tool)
        coords = []
        if flag_tool == "rectangle" or flag_tool == "box":
            for rect in iv_object.rect_list:
                pts = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3]),(rect[0], rect[1])]
                coords.append(pts)
        elif flag_tool == "polygon" or flag_tool == "free":
            for poly in iv_object.poly_list:
                pts = np.array(poly, dtype=np.int32)
                coords.append(pts)

    ## create mask
    mask = {"label": label,
            "include": include,
            "coords": str(coords)}

    ## show image with window control
    if flag_show:
        overlay = np.zeros(image.shape, np.uint8) # make overlay
        overlay[:,:,2] = 200 # start with all-red overlay
        mask_bin = _create_mask_bin(image, coords)
        mask_bool = np.array(mask_bin, dtype=bool)
        if include:
            overlay[mask_bool,1], overlay[mask_bool,2] = 200, 0
        else:
            overlay[np.invert(mask_bool),1], overlay[np.invert(mask_bool),2] = 200, 0
        mask_overlay = cv2.addWeighted(image, .7, overlay, 0.5, 0)
        show_image(mask_overlay)

    ## overwrite if directory
    if obj_input.__class__.__name__ == "container":
        if obj_input.dirpath:
            if skip == False:
                mask_path = os.path.join(obj_input.dirpath, "masks.yaml")
                if os.path.isfile(mask_path):
                    masks = _load_yaml(mask_path)
                else:
                    masks = {}
                masks[label] = mask
                _save_yaml(masks, mask_path)

    ## return mask
    if obj_input.__class__.__name__ == "container":
        obj_input.masks[label] = mask
    else:
        return mask


class mask_data(object):
    """ This is a mask-data object where the image and other data 
      is stored that can be passed on between pype-steps
    
    Parameters
    ----------

    mask_list: list
        list of drawn masks (clockwise cornerpoints of rectangles or polygons)
    mask_overlay: array
        input image with drawn mask contours
    mask_bin: array
        binary mask 
    mask_bool: array
        boolean mask
    label: str
        mask label
    include: bool
        flag whether mask area is included or excluded
        
    """    
    def __init__(self, coords, mask_overlay, mask_bin, mask_bool, label, include):
        self.label = label
        self.include = include
        self.coords = coords
        self.mask_overlay = mask_overlay
        self.mask_bin = mask_bin
        self.mask_bool = mask_bool

