#%%
import cv2
import copy
import numpy as np
import sys 

from phenopype.settings import colours
from phenopype.utils import show_img
from phenopype.utils_lowlevel import _auto_line_thickness, _image_viewer

#%%

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
        
    ## load image
    if isinstance(obj_input, str):
        image = cv2.imread(obj_input)  
    elif obj_input.__class__.__name__ == "pype_container":
        image = obj_input.image_mod
        
    ## kwargs
    label = kwargs.get("label","mask1")
    max_dim = kwargs.get("max_dim", 1000)
    include = kwargs.get("include",True)
    flag_show = kwargs.get("show",False)
    flag_tool = kwargs.get("tool","rectangle")
    flag_overwrite = kwargs.get("overwrite", False)
    
    ## check if mask exists 
    if obj_input.__class__.__name__ == "pype_container":
        if label in obj_input.mask_binder:
            if not flag_overwrite:
                return obj_input
            else:
                pass
    
    ## method
    iv_object = _image_viewer(image, 
                              mode="interactive", 
                              max_dim = max_dim, 
                              tool=flag_tool)
    
    ## draw masks into black canvas
    zeros = np.zeros(image.shape[0:2], np.uint8)
    if flag_tool == "rectangle" or flag_tool == "box":
        for rect in iv_object.rect_list:
            pts = np.array(((rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])), dtype=np.int32)
            mask_bin = cv2.fillPoly(zeros, [pts], colours.white)
    elif flag_tool == "polygon" or flag_tool == "free":
        for poly in iv_object.poly_list:
            pts = np.array(poly, dtype=np.int32)
            mask_bin = cv2.fillPoly(zeros, [pts], colours.white)

    ## create boolean mask
    mask_bool = np.array(mask_bin, dtype=bool)
    if include == False:
        mask_bool = np.invert(mask_bool)
        
    ## create overlay
    overlay = np.zeros(image.shape, np.uint8) # make overlay
    overlay[:,:,2] = 200 # start with all-red overlay
    overlay[mask_bool,1] = 200   
    overlay[mask_bool,2] = 0   
    mask_overlay = cv2.addWeighted(image, .7, overlay, 0.5, 0)
    
    ## show image
    if flag_show:
        show_img(mask_overlay)
    if flag_tool == "rectangle" or flag_tool == "box":
        mask_list = iv_object.rect_list
    elif flag_tool == "polygon" or flag_tool == "free":
        mask_list = iv_object.poly_list
        
    ## create mask data object
    MO = mask_data(mask_list=mask_list, 
                     mask_overlay=mask_overlay, 
                     mask_bin=mask_bin, 
                     mask_bool = mask_bool, 
                     label=label,
                     include=include)
    
    # MO.__class__.__name__ = label
    
    ## window control
    if cv2.waitKey() == 13:
        cv2.destroyAllWindows()
    elif cv2.waitKey() == 27:
        cv2.destroyAllWindows()
        sys.exit("Esc: exit phenopype process")    
    
    ## return
    if obj_input.__class__.__name__ == "pype_container":
        obj_input.mask_binder[label] = MO
        return obj_input
    else:
        return MO

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
        
    ## load image
    if isinstance(obj_input, str):
        image = cv2.imread(obj_input)
    elif obj_input.__class__.__name__ == "pype_container":
        image = obj_input.image_mod
        mask_binder = obj_input.mask_binder
        
    ## kwargs
    mask_filter = kwargs.get("filter",mask_binder)
    line_thickness = kwargs.get("line_thickness", _auto_line_thickness(image))
    colour = eval("colours." + kwargs.get("colour", "green"))

    ## draw masks from mask obect    
    for key, value in mask_binder.items():
        if key in mask_filter:
            MO = value
            for (rx1, ry1, rx2, ry2) in MO.mask_list:
                cv2.rectangle(image, (rx1,ry1), (rx2,ry2), colour, line_thickness)
                
    ## return
    if obj_input.__class__.__name__ == "pype_container":
        obj_input.image_mod = image
        return obj_input


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
    def __init__(self, mask_list, mask_overlay, mask_bin, mask_bool, label, include):
        self.label = label
        self.include = include
        self.mask_overlay = mask_overlay
        self.mask_bin = mask_bin
        self.mask_bool = mask_bool
        self.mask_list = mask_list

        

