import cv2
import numpy as np

from phenopype.utils_lowlevel import _image_viewer
from phenopype.utils import show_img

#%% colours

green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
black = (0,0,0)
white = (255,255,255)



#%% methods

def create_mask(image, **kwargs):
    """Mask maker method to draw rectangle or polygon mask onto image.
    
    Parameters
    ----------        
    
    include: bool (default: True)
        determine whether resulting mask is to include or exclude objects within
    label: str (default: "area1")
        passes a label to the mask
    mode: str (default: "rectangle")
        zoom into the scale with "rectangle" or "polygon".
        
    """
        
    ## load image
    if isinstance(image, str):
        image = cv2.imread(image)  
    if len(image.shape)==2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    label = kwargs.get("label","area 1")
    max_dim = kwargs.get("max_dim",1980)
    include = kwargs.get("include",True)
    
    flag_show = kwargs.get("show",False)
    flag_tool = kwargs.get("tool","rectangle")

    print("\nMark the outline of your arena, i.e. what you want to include in the image analysis by left clicking, finish with enter.")

    iv_object = _image_viewer(image, mode="interactive", max_dim = max_dim, tool=flag_tool)
    
    zeros = np.zeros(image.shape[0:2], np.uint8)
    
    if flag_tool == "rectangle" or flag_tool == "box":
        for rect in iv_object.rect_list:
            pts = np.array(((rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])), dtype=np.int32)
            mask_bin = cv2.fillPoly(zeros, [pts], white)
    elif flag_tool == "polygon" or flag_tool == "free":
        for poly in iv_object.poly_list:
            pts = np.array(poly, dtype=np.int32)
            mask_bin = cv2.fillPoly(zeros, [pts], white)

    if include == False:
        mask_bool = np.invert(mask_bin)
    mask_bool = np.array(mask_bin, dtype=bool)

    overlay = np.zeros(image.shape, np.uint8) # make overlay
    overlay[:,:,2] = 200 # start with all-red overlay
    overlay[mask_bool,1] = 200   
    overlay[mask_bool,2] = 0   
    overlay = cv2.addWeighted(image, .7, overlay, 0.5, 0)
    
    if flag_show:
        show_img(overlay)
    if flag_tool == "rectangle" or flag_tool == "box":
        mask_list = iv_object.rect_list
    elif flag_tool == "polygon" or flag_tool == "free":
        mask_list = iv_object.poly_list

    class mask_object:
        def __init__(self, label, overlay, zeros, mask_bool, mask_list):
            self.label = label
            self.overlay = overlay
            self.mask_bin = zeros
            self.mask_bool = mask_bool
            self.mask_list = mask_list
    
    return mask_object(label, overlay, zeros, mask_bool, mask_list)  # Create an empty record