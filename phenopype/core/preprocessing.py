#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours
from phenopype.utils import load_image, load_yaml, load_meta_data, show_image, show_yaml, save_image, save_yaml
from phenopype.utils_lowlevel import _image_viewer

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
        image = obj_input.image
        
    # image, flag_input = _load_image(obj_input)
    # if obj_input.flag_pp_proj:
    #     attr = load_yaml(os.path.join(obj_input.dirpath, "attributes.yaml"))

    ## check if mask exists 
    if obj_input.__class__.__name__ == "container":
        if obj_input.masks:
            if "masks" in attr:
                if label in attr["masks"] and not flag_overwrite:
                    skip = True
                    coords = eval(attr["masks"][label]["coords"])
        elif label in obj_input.mask_binder:
            if not flag_overwrite:
                return obj_input

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

    ## create binary mask and boolean mask
    mask_bin = np.zeros(image.shape[0:2], np.uint8)
    for sub_coords in coords:
        cv2.fillPoly(mask_bin, [np.array(sub_coords, dtype=np.int32)], colours.white)
    mask_bool = np.array(mask_bin, dtype=bool)

        
    ## create overlay
    overlay = np.zeros(image.shape, np.uint8) # make overlay
    overlay[:,:,2] = 200 # start with all-red overlay
    if include:
        overlay[mask_bool,1], overlay[mask_bool,2] = 200, 0
    else:
        overlay[np.invert(mask_bool),1], overlay[np.invert(mask_bool),2] = 200, 0
    mask_overlay = cv2.addWeighted(image, .7, overlay, 0.5, 0)

    ## create mask data object
    MO = mask_data(coords=coords, 
                     mask_overlay=mask_overlay, 
                     mask_bin=mask_bin, 
                     mask_bool = mask_bool, 
                     label=label,
                     include=include)

    ## show image with window control
    if flag_show:
        show_image(mask_overlay)
    if cv2.waitKey() == 13:
        cv2.destroyAllWindows()
    elif cv2.waitKey() == 27:
        cv2.destroyAllWindows()
        sys.exit("Esc: exit phenopype process")    

    ## if pp-project, add to attributes
    if obj_input.__class__.__name__ == "container":
        mask = ordereddict([("include",include),("coords",str(coords))])
        if not "masks" in attr:
            attr["masks"] = {}
        attr["masks"][label] = mask
        save_yaml(attr, os.path.join(obj_input.dirpath, "attributes.yaml"))

    ## return mask
    if obj_input.__class__.__name__ == "pype_container":
        obj_input.mask_binder[label] = MO
        return obj_input
    else:
        return MO


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

