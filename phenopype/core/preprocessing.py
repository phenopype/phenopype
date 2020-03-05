#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from datetime import datetime
from math import sqrt
import numpy.ma as ma

from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours
from phenopype.utils import load_image, load_meta_data, show_image, save_image
from phenopype.utils_lowlevel import _image_viewer, _create_mask_bin, _equalize_histogram, _contours_arr_tup
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

    ## check if exists
    while True:
        if not df_masks.__class__.__name__ == "NoneType" and flag_overwrite == False:
            df_masks = df_masks[df_masks.columns.intersection(["mask", "include", "coords"])]
            if label in df_masks["mask"].values:
                print("- mask with label " + label + " already created (overwrite=False)")
                break
        elif not df_masks.__class__.__name__ == "NoneType" and flag_overwrite == True:
            df_masks = df_masks[df_masks.columns.intersection(["mask", "include", "coords"])]
            if label in df_masks["mask"].values:
                df_masks = df_masks.drop(df_masks[df_masks["mask"] == label].index)
                print("- create mask (overwriting)")
                pass
        elif df_masks.__class__.__name__ == "NoneType":
            print("- create mask")
            df_masks = pd.DataFrame(columns=["mask", "include", "coords"])
            pass

        ## method
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
                df_masks = df_masks.append({"mask": label, "include": include, "coords": str(coords)}, 
                                ignore_index=True, sort=False)
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

    ## method
    out = _image_viewer(image, tool="scale")
    points = out.scale_coords
    distance_px = int(sqrt(((points[0][0]-points[1][0])**2)+((points[0][1]-points[1][1])**2)))
    entry = enter_data(out.canvas, columns="length")
    distance_mm = int(entry["length"][0])
    px_mm_ratio = int(distance_px / distance_mm)

    ## create template for image registration
    if flag_template:
        out = _image_viewer(image, tool="template")
        template = image[out.rect_list[0][1]:out.rect_list[0][3],out.rect_list[0][0]:out.rect_list[0][2]]
    else:
        template = None

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return px_mm_ratio, template
    elif obj_input.__class__.__name__ == "container":
        obj_input.scale_px_mm_ratio = px_mm_ratio
        obj_input.scale_template = template



def find_scale(obj_input, **kwargs):

    """Find scale from a defined template inside an image and update pixel 
    ratio. Image registration is run by the "AKAZE" algorithm 
    (http://www.bmva.org/bmvc/2013/Papers/paper0013/abstract0013.pdf). 
    Future implementations will include more algorithms to select from.
    Prior to running detect_scale, measure_scale and make_scale_template 
    have to be run once to pass on reference scale size and template of 
    scale reference card (gets passed on internally by calling detect_scale 
    from the same instance of scale_maker.
    
    Parameters
    -----------
    obj_input: array or phenopype-container
        input for processing
    resize: num (optional, default: 1 or 0.5 for images with diameter > 5000px)
        resize image to speed up detection process (WARNING: too low values may result in poor detection results or even crashes)
    show: bool (optional, default: False)
        show result of scale detection procedure on current image  
    """

    ## kwargs
    flag_overwrite = kwargs.get("overwrite", False)
    flag_equalize = kwargs.get('equalize', True)
    min_matches = kwargs.get('min_matches', 10)
    px_mm_ratio_old = kwargs.get("px_mm_ratio", None)
    template = kwargs.get("template", None)

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image_copy
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "scale_template_px_mm_ratio"):
            px_mm_ratio_old = obj_input.scale_template_px_mm_ratio
        if hasattr(obj_input, "scale_template"):
            template = obj_input.scale_template
        if hasattr(obj_input, "df_masks"):
            df_masks = copy.deepcopy(obj_input.df_masks)
        else:
            df_masks = pd.DataFrame(columns=["mask", "include", "coords"])

    ## check if all info has been prvided
    while True:
        if any([px_mm_ratio_old.__class__.__name__ == "NoneType", 
                template.__class__.__name__ == "NoneType"]):
            print("- scale information missing - abort")
            break
        if hasattr(obj_input, "scale_current_px_mm_ratio") and not flag_overwrite:
            scale_current_px_mm_ratio = obj_input.scale_current_px_mm_ratio
            print("- scale already detected (overwrite=False)")
            break    
        elif hasattr(obj_input, "scale_current_px_mm_ratio") and flag_overwrite:
            print(" - detecting scale (overwriting)")
            pass

        ## if image diameter bigger than 2000 px, then automatically resize
        if (image.shape[0] + image.shape[1])/2 > 5000:
            resize_factor = kwargs.get('resize', 0.5)
            warnings.warn("large image - resizing by factor " + str(resize_factor) + " to avoid slow image registration")
        else:
            resize_factor = kwargs.get('resize', 1)
        image_resized = cv2.resize(image, (0,0), fx=1*resize_factor, fy=1*resize_factor) 

        ## method
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(template, None)
        kp2, des2 = akaze.detectAndCompute(image_resized, None)       
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        matches = matcher.knnMatch(des1, des2, 2)

        # keep only good matches
        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # find and transpose coordinates of matches
        if len(good) >= min_matches:
            ## find homography betweeen detected keypoints
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            ## transform boundary box of template
            rect_old = np.array([[[0, 0]], 
                                   [[0, template.shape[0]]],  
                                   [[template.shape[1], template.shape[0]]], 
                                   [[template.shape[1], 0]]], dtype= np.float32)
            rect_new = cv2.perspectiveTransform(rect_old,M)/resize_factor

            # calculate template diameter
            rect_new = rect_new.astype(np.int32)
            (x,y),radius = cv2.minEnclosingCircle(rect_new)
            diameter_new = (radius * 2)

            # calculate transformed diameter
            rect_old = rect_old.astype(np.int32)
            (x,y),radius = cv2.minEnclosingCircle(rect_old)
            diameter_old = (radius * 2)

            ## calculate ratios
            diameter_ratio = (diameter_new / diameter_old)
            px_mm_ratio_new = round(diameter_ratio * px_mm_ratio_old, 1)

            ## feedback
            print("---------------------------------------------------")
            print("Reference card found with %d keypoint matches:" % len(good))
            print("template image has %s pixel per mm." % (px_mm_ratio_old))
            print("current image has %s pixel per mm." % (px_mm_ratio_new))
            print("= %s %% of template image." % round(diameter_ratio * 100, 3))
            print("---------------------------------------------------")

            ## create mask from new coordinates
            coords = _contours_arr_tup(rect_new)
            coords.append(coords[0])
            if "scale" in df_masks['mask'].values:
                df_masks.drop(df_masks.loc[df_masks["mask"]=="scale"].index, inplace=True)
            row_scale = pd.DataFrame({"mask": "scale", "include": False, "coords": str([coords])}, index=[0])
            row_scale = pd.concat([pd.concat([df_image_data]*len(row_scale)).reset_index(drop=True), 
                        row_scale.reset_index(drop=True)], axis=1)
            df_masks = df_masks.append(row_scale, sort=False)  
            scale_current_px_mm_ratio = px_mm_ratio_new
            break

    ## rectangle coords of scale in image
    rect_new = eval(df_masks.loc[df_masks["mask"]=="scale", "coords"].reset_index(drop=True)[0])

    ## do histogram equalization
    if flag_equalize:
        detected_rect_mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(detected_rect_mask, [np.array(rect_new)], colours["white"]) 
        (rx,ry,rw,rh) = cv2.boundingRect(np.array(rect_new))
        detected_rect_mask =  ma.array(data=image[ry:ry+rh,rx:rx+rw], mask = detected_rect_mask[ry:ry+rh,rx:rx+rw])
        image = _equalize_histogram(image, detected_rect_mask, template)

    
    ## merge with existing image_data frame
    df_image_data["px_mm_ratio"] = scale_current_px_mm_ratio

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        image = cv2.polylines(image,[rect_new], True,colours["red"],5, cv2.LINE_AA)
        return scale_current_px_mm_ratio, image
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_masks = df_masks
        obj_input.scale_current_px_mm_ratio = scale_current_px_mm_ratio
        if flag_equalize:
            obj_input.image_copy = image


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
    

