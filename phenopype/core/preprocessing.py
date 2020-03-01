#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from datetime import datetime
from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours
from phenopype.utils import load_image, load_meta_data, show_image, save_image
from phenopype.utils_lowlevel import _image_viewer, _create_mask_bin #, _load_masks
from phenopype.utils_lowlevel import _load_yaml, _show_yaml, _save_yaml, _yaml_file_monitor, _auto_line_width

#%% functions

def create_mask(obj_input, **kwargs):
    """Mask maker method to draw rectangle or polygon mask onto image.
    
    Parameters
    ----------        
    
    include (optional): bool (default: True)
        determine whether resulting mask is to include or exclude objects within
    label(optional): str (default: "mask1")
        passes a label to the mask
    max_dim (optional): int (default: 1000)
        maximum dimension of the window along any axis in pixel
    overwrite (optional): bool (default: False)
        if working using a container, or from a phenopype project directory, should
        existing masks with the same label be overwritten
    show (otpional): bool (default: False)
        should the drawn mask be shown as an overlay on the image
    tool (optional): str (default: "rectangle")
        draw mask by draging a rectangle (option: "rectangle") or by settings 
        points for a polygon (option: "polygon").
        
    """

    ## kwargs
    df_image_data = kwargs.get("df_image_data", None)
    flag_overwrite = kwargs.get("overwrite", False)

    label = kwargs.get("label","mask1")
    max_dim = kwargs.get("max_dim", 1000)
    include = kwargs.get("include",True)
    flag_show = kwargs.get("show",False)
    flag_tool = kwargs.get("tool","rectangle")
    
    ## load image
    df_masks = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"}, index=[0])
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_masks"):
            df_masks = copy.deepcopy(obj_input.df_masks)
    else:
        warnings.warn("wrong input format.")
        return

    ## more kwargs
    line_width = kwargs.get("line_width", _auto_line_width(image))

    ## check if exists
    if not df_masks.__class__.__name__ == "NoneType" and flag_overwrite == False:
        if label in df_masks["mask"].values:
            print("- mask with label " + label + " already created (overwrite=False)")
            return
    elif not df_masks.__class__.__name__ == "NoneType" and flag_overwrite == True:
        if label in df_masks["mask"].values:
            df_masks.drop(df_masks[df_masks["mask"] == label].index, inplace=True)
            print("- create mask (overwriting)")
    elif df_masks.__class__.__name__ == "NoneType":
        print("- create mask")
        df_masks = pd.DataFrame(columns=["mask", "include", "coords"])

    ## create mask
    out = _image_viewer(image, mode="interactive", 
                              max_dim = max_dim, 
                              tool=flag_tool)
    coords = out.point_list
    
    ## abort
    if not out.done:
        if obj_input.__class__.__name__ == "ndarray":
            warnings.warn("terminated mask creation")
            return 
        elif obj_input.__class__.__name__ == "container":
            print("- terminated mask creation")
            return True

    ## create df
    if len(coords) > 0:
        for points in coords:
            mask = {"mask": label,
                    "include": include,
                    "coords": str(points)}
            df_masks = df_masks.append(mask, ignore_index=True, sort=False)
        df_masks = pd.concat([pd.concat([df_image_data]*len(df_masks)).reset_index(drop=True), 
                                df_masks.reset_index(drop=True)], axis=1)
    else:
        warnings.warn("zero coordinates - redo mask!")

    ## return
    if obj_input.__class__.__name__ == "ndarray":
            return image, df_masks
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_masks = df_masks
        obj_input.canvas = image



# def enter_data(obj_input, **kwargs):
    
#     print(kwargs)
    # def _keyboard_entry(event, x, y, flags, params):
    #     pass
    
    # done = False
    # if ID.__class__.__name__ == "NoneType":
    #     ID = ""
    #     while not done or ID == "":
            
    #         cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
    #         cv2.setMouseCallback("phenopype", _keyboard_entry)
    #         k = cv2.waitKey(1)
    
    #         if k > 0 and k != 8 and k != 13 and k != 27:
    #             ID = ID + chr(k)
    #         elif k == 8:
    #             ID = ID[0:len(ID)-1]
    #         image_warning = copy.deepcopy(out.image_copy)
    #         cv2.putText(image_warning, "Enter ID: " + ID, (int(image_warning.shape[0]//10),int(image_warning.shape[1]/3)), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, _auto_text_size(image)+10, colours["red"], _auto_text_width(image)*10, cv2.LINE_AA)
            
    #         cv2.imshow("phenopype", image_warning)
    #         if k == 27:
    #             cv2.destroyWindow("phenopype")
    #             break
    #         elif k == 13:
    #             if not ID =="":
    #                 done = True
    #                 cv2.destroyWindow("phenopype")
    #                 break



def invert_image(obj_input, **kwargs):
    """
    

    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    
    ## load image and check if pp-project
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image

    ## method
    image = cv2.bitwise_not(image)

    ## return 
    if obj_input.__class__.__name__ == "container":
        obj_input.image = image
    else:
        return image
    

