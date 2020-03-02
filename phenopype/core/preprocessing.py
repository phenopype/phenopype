#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from datetime import datetime
from math import sqrt
from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours
from phenopype.utils import load_image, load_meta_data, show_image, save_image
from phenopype.utils_lowlevel import _image_viewer, _create_mask_bin #, _load_masks
from phenopype.utils_lowlevel import _load_yaml, _show_yaml, _save_yaml, _yaml_file_monitor, _auto_line_width, _auto_text_size, _auto_text_width

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

    ## mask df only
    df_masks = df_masks[df_masks.columns.intersection(["mask", "include", "coords"])]

    ## check if exists
    while True:
        if not df_masks.__class__.__name__ == "NoneType" and flag_overwrite == False:
            if label in df_masks["mask"].values:
                print("- mask with label " + label + " already created (overwrite=False)")
                break
        elif not df_masks.__class__.__name__ == "NoneType" and flag_overwrite == True:
            if label in df_masks["mask"].values:
                df_masks.drop(df_masks[df_masks["mask"] == label].index, inplace=True)
                print("- create mask (overwriting)")
                pass
        elif df_masks.__class__.__name__ == "NoneType":
            print("- create mask")
            df_masks = pd.DataFrame(columns=["mask", "include", "coords"])
            pass

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
        else:
            warnings.warn("zero coordinates - redo mask!")
        break

    ## merge with existing image_data frame
    df_masks = pd.concat([pd.concat([df_image_data]*len(df_masks)).reset_index(drop=True), 
                            df_masks.reset_index(drop=True)], axis=1)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
            return image, df_masks
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_masks = df_masks
        obj_input.canvas = image



def create_scale(obj_input, **kwargs):
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
    flag_template = kwargs.get("template", True)

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
    df_scale = pd.DataFrame({}, index=[0])

    out = _image_viewer(image, tool="scale")
    points = out.scale_coords
    df_scale["distance_px"] = int(sqrt(((points[0][0]-points[1][0])**2)+((points[0][1]-points[1][1])**2)))
    df = enter_data(out.canvas, columns="length")
    df_scale["distance_mm"] = df["length"][0]
    
    if flag_template:
        out = _image_viewer(image, tool="template")
        template = image[out.rect_list[0][1]:out.rect_list[0][3],out.rect_list[0][0]:out.rect_list[0][2]]

    if not flag_template:
        return df_scale
    else:
        return df_scale, template

def enter_data(obj_input, **kwargs):
    """
    

    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ## kwargs
    df_image_data = kwargs.get("df_image_data", None)
    flag_overwrite = kwargs.get("overwrite", False)
    columns = kwargs.get("columns", "ID")
    columns = columns.replace(" ","")
    columns = columns.split(",")
    
    ## load image
    df_other_data = pd.DataFrame({}, index=[0])
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        image_copy = copy.deepcopy(obj_input)
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"}, index=[0])
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.canvas)
        image_copy = copy.deepcopy(obj_input.canvas)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_other_data"):
            df_other_data = obj_input.df_other_data
    else:
        warnings.warn("wrong input format.")
        return

    ## more kwargs
    label_size = kwargs.get("label_size", _auto_text_size(image)*2)
    label_width = kwargs.get("label_width", _auto_text_width(image)*2)
    label_col = kwargs.get("label_col", "red")

    ## keyboard listener
    def _keyboard_entry(event, x, y, flags, params):
        pass

    ## check if exists
    while True:
        for col in columns:
            if col in df_other_data and flag_overwrite == False:
                print("- column " + col + " already created (overwrite=False)")
                continue
            elif col in df_other_data and flag_overwrite == True:
                df_other_data.drop([col], inplace=True, axis=1)
                print("- add column " + col + " (overwriting)")
                pass
            else:
                print("- add column " + col)
                pass

            entry = ""
            while True or entry == "":

                cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("phenopype", _keyboard_entry)

                k = cv2.waitKey(1)
                image = copy.deepcopy(image_copy)

                if k > 0 and k != 8 and k != 13 and k != 27:
                    entry = entry + chr(k)
                elif k == 8:
                    entry = entry[0:len(entry)-1]

                cv2.putText(image, "Enter " + col + ": " + entry, (int(image.shape[0]//10),int(image.shape[1]/3)), 
                        cv2.FONT_HERSHEY_SIMPLEX, label_size, colours[label_col],label_width, cv2.LINE_AA)
                cv2.imshow("phenopype", image)


                if k == 27:
                    cv2.destroyWindow("phenopype")
                    break
                    return True
                elif k == 13:
                    if not entry =="":
                        cv2.destroyWindow("phenopype")
                        break
            df_other_data[col] = entry
        break

    ## drop unspecified columns
    for col in list(df_other_data):
        if col not in columns:
            df_other_data.drop(columns=col, inplace = True)

    ## merge with existing image_data frame
    df_image_data = pd.concat([pd.concat([df_image_data]*len(df_other_data)).reset_index(drop=True), 
                            df_other_data.reset_index(drop=True)], axis=1)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
            return df_image_data
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_image_data = df_image_data
        obj_input.df_other_data = df_other_data


def find_scale(obj_input, **kwargs):

    """Find scale from a defined template inside an image and update pixel ratio. Image registration is run by the "AKAZE" algorithm 
    (http://www.bmva.org/bmvc/2013/Papers/paper0013/abstract0013.pdf). Future implementations will include more algorithms to select from.
    Prior to running detect_scale, measure_scale and make_scale_template have to be run once to pass on reference scale size and template of 
    scale reference card (gets passed on internally by calling detect_scale from the same instance of scale_maker.
    
    Parameters
    -----------
    target_image: str or array
        absolute or relative path to OR numpy array of targeted image that contains the scale reference card
    resize: num (optional, default: 1 or 0.5 for images with diameter > 5000px)
        resize image to speed up detection process (WARNING: too low values may result in poor detection results or even crashes)
    show: bool (optional, default: False)
        show result of scale detection procedure on current image  
    """

    ## load image
    df_scale = pd.DataFrame({}, index=[0])
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image_copy
        if hasattr(obj_input, "df_scale"):
            df_scale = obj_input.df_scale


    ## kwargs
    min_matches = kwargs.get('min_matches', 10)
    flag_show = kwargs.get('show', False)
    flag_equalize = kwargs.get('equalize', False)
    # template = kwargs.get("template", load_image(obj_input.template_path))
        
    if not len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    ## if image diameter bigger than 2000 px, then automatically resize
    if (image.shape[0] + image.shape[1])/2 > 2000:
        resize_factor = kwargs.get('resize', 0.5)
        warnings.warn("large image - resizing by factor " + str(resize_factor))
    else:
        resize_factor = kwargs.get('resize', 1)
    image = cv2.resize(image, (0,0), fx=1*resize_factor, fy=1*resize_factor) 

    # =============================================================================
    # AKAZE detector
    # =============================================================================     
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(template, None)
    kp2, des2 = akaze.detectAndCompute(image, None)       
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.knnMatch(des1, des2, 2)

    # keep only good matches
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    nkp = len(good)
    
    # find and transpose coordinates of matches
    if nkp >= min_matches:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        box_mask = np.zeros(template.shape[0:2], np.uint8)
        box_mask.fill(255)
        ret, contours, hierarchy = cv2.findContours(box_mask,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_TC89_L1)
        box = contours[0].astype(np.float32)
        rect  = cv2.perspectiveTransform(box,M).astype(np.int32)

        ## draw onto image
        image = cv2.polylines(image,[rect],True,colours["red"],5, cv2.LINE_AA)

        # update current scale using reference
        rect = rect/resize_factor
        rect = np.array(rect, dtype=np.int32)
        (x,y),radius = cv2.minEnclosingCircle(rect)
        detected_diameter 
        
        rect_old = np.array(box, dtype=np.int32)
        (x,y),radius = cv2.minEnclosingCircle(rect_old)        
        reference_diameter = (radius * 2)
        
        diff_ratio = (detected_diameter / reference_diameter)
        current = round(scale_ratio * diff_ratio,1)


        ## resize target image back to original size
        target_image = cv2.resize(target_image, (0,0), fx=1/resize_factor, fy=1/resize_factor) 
        
        # create mask of detected scale reference card
        zeros = np.zeros(target_image.shape[0:2], np.uint8)
        mask_bin = cv2.fillPoly(zeros, [np.array(rect)], colours.white)       
        detected_mask = np.array(mask_bin, dtype=bool)
        
        # cut out target reference card
        (rx,ry,w,h) = cv2.boundingRect(rect)
        target_detected = target_image_original[ry:ry+h,rx:rx+w]
        
        print("\n")
        print("---------------------------------------------------")
        print("Reference card found with %d keypoint matches:" % nkp)
        print("current image has %s pixel per mm." % (scale_ratio))
        print("= %s %% of template image." % round(diff_ratio * 100,3))
        print("---------------------------------------------------")
        print("\n")
        
        ## do histogram equalization
        if flag_equalize:
            _equalize()

        ## show results
        if flag_show:
            show_img([template_image, target_image, target_image_corrected])
            
        return target_image_corrected, (detected_mask, "scale-detected", False), current       

    
    else:
        print("\n")
        print("---------------------------------------------------")
        print("Reference card not found - only %d/%d keypoint matches" % (nkp, min_matches))
        print("---------------------------------------------------")
        print("\n")
        
        return "no current scale", "no scale mask"




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
    

