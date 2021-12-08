#%% modules

import cv2, copy
import numpy as np
import pandas as pd
from dataclasses import make_dataclass

from math import sqrt as _sqrt
import numpy.ma as ma

from phenopype.settings import colours, flag_verbose, _image_viewer_arg_list
from phenopype.utils_lowlevel import (
    _auto_text_width, 
    _auto_text_size, 
    _convert_arr_tup_list, 
    _equalize_histogram, 
    _resize_image, 
    _ImageViewer, 
    _load_previous_annotation,
    _drop_dict_entries,
    _update_settings,
    )
import phenopype.core.segmentation as segmentation

#%% functions


def blur(
    image,
    kernel_size=5,
    method="averaging",
    sigma_color=75,
    sigma_space=75,
    **kwargs
):
    """
    Apply a blurring algorithm to an image.

    Parameters
    ----------
    image: array 
        input image
    kernel_size: int, optional
        size of the blurring kernel (has to be odd - even numbers will be ceiled)
    method: {averaging, gaussian, median, bilateral} str, optional
        blurring algorithm
    sigma_colour: int, optional
        for 'bilateral'
    sigma_space: int, optional
        for 'bilateral'

    Returns
    -------
    image : ndarray 
        blurred image
    """
    
	# =============================================================================
	# setup 
    
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
        if flag_verbose:
            print("- even kernel size supplied, adding 1 to make odd")
            
	# =============================================================================
	# execute
    
    if method == "averaging":
        blurred = cv2.blur(image, (kernel_size, kernel_size))
    elif method == "gaussian":
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        blurred = cv2.medianBlur(image, kernel_size)
    elif method == "bilateral":
        blurred = cv2.bilateralFilter(image, kernel_size, sigma_color, sigma_space)

	# =============================================================================
	# return
    
    return blurred


def create_mask(
    image,
    include=True,
    tool="rectangle",
    **kwargs
):           
    """
    Mask an area.

    Parameters
    ----------
    image: array 
        input image
    include : bool, optional
        include or exclude area inside mask
    tool : {"rectangle","polygon"} str, optional
        Type of mask tool to be used. The default is "rectangle".

    Returns
    -------
    annotations: dict
        phenopype annotation containing mask coordinates

    """
    
    # =============================================================================
    # setup
    
    annotation_previous = kwargs.get("annotation_previous")
    
    # =============================================================================
    # retain settings

    ## retrieve settings from args
    local_settings  = _drop_dict_entries(locals(),
        drop=["image","kwargs","annotation_previous"])

    ## retrieve update IV settings and data from previous annotations  
    IV_settings = {}     
    if annotation_previous:       
        IV_settings["ImageViewer_previous"] =_load_previous_annotation(
            annotation_previous = annotation_previous, 
            components = [
                ("data","coord_list"),
                ])            
        
    ## update local and IV settings from kwargs
    if kwargs:
        _update_settings(kwargs, local_settings, IV_settings)

	# =============================================================================
	# execute
    
    out = _ImageViewer(image=image, 
                        tool=tool, 
                        **IV_settings)
    
    ## check if tasks completed successfully
    if not out.done:
        print("- didn't finish: redo mask")
        return 
    if not out.coord_list:
        print("- zero coordinates: redo mask")
        return 
        
	# =============================================================================
	# assemble results
    
    annotation = {
        "info": {
            "annotation_type": "mask",
            "pp_function": "create_mask",
        },
        "settings": local_settings,
        "data": {
            "include": include,
            "n_masks": len(out.coord_list),
            "coord_list": out.coord_list,
        }
    }
    
	# =============================================================================
	# return
    
    return annotation

    
def detect_shape(
    image,
    include=True,
    shape="circle",
    resize=1,
    circle_args={
        "dp":1,
        "min_dist":50,
        "param1":200,
        "param2":100,
        "min_radius":0,
        "max_radius":0
        },
    **kwargs
):
    """
    
    Detects geometric shapes in a single channel image (currently only circles 
    are implemented) and converts boundary contours to a mask to include or exclude
    parts of the image. Depending on the object type, different settings apply.

    Parameters
    ----------
    image : ndarray
        input image (single channel).
    include : bool, optional
        should the resulting mask include or exclude areas. The default is True.
    shape : str, optional
        which geometric shape to be detected. The default is "circle".
    resize : (0.1-1) float, optional
        resize factor for image (some shape detection algorithms are slow if the 
        image is very large). The default is 1.
    circle_args : dict, optional
        A set of options for circle detection (for details see
        https://docs.opencv.org/3.4.9/dd/d1a/group__imgproc__feature.html ):
            
            - dp: inverse ratio of the accumulator resolution to the image ressolution
            - minDist: minimum distance between the centers of the detected circles
            - param1: higher threshold passed to the canny-edge detector
            - param2: accumulator threshold - smaller = more false positives
            - min_radius: minimum circle radius
            - max_radius: maximum circle radius
            
        The default is:
            
        .. code-block:: python
        
            {
                "dp":1,
                 "min_dist":50,
                 "param1":200,
                 "param2":100,
                 "min_radius":0,
                 "max_radius":0
                 }

    Returns
    -------
    annotations: dict
        phenopype annotation containing mask coordinates

    """

    

    # =============================================================================
    # setup
    
    if len(image.shape) == 3:
        image = decompose_image(image, "gray")
    image_resized = _resize_image(image, resize)
    
    
    # =============================================================================
    # retain settings

    ## retrieve settings from args
    local_settings  = _drop_dict_entries(
        locals(), drop=["image","kwargs", "image_resized"])
        
    ## update settings from kwargs
    if kwargs:
        _update_settings(kwargs, local_settings)
        
        
    # =============================================================================
    # execute 
    
    if shape=="circle":
        circles = cv2.HoughCircles(image_resized, 
                                   cv2.HOUGH_GRADIENT, 
                                   dp=max(int(circle_args["dp"]*resize),1), 
                                   minDist=int(circle_args["min_dist"]*resize),
                                   param1=int(circle_args["param1"]*resize), 
                                   param2=int(circle_args["param2"]*resize),
                                   minRadius=int(circle_args["min_radius"]*resize), 
                                   maxRadius=int(circle_args["max_radius"]*resize))
    
        ## output conversion
        if circles is not None:
            coords = []
            for idx, circle in enumerate(circles[0]):
                x,y,radius = circle/resize
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask = cv2.circle(mask, (x,y), radius, 255, -1)
                mask_contours = segmentation.detect_contour(
                    mask,
                    retrieval="ext", 
                    approximation="KCOS", 
                    verbose=False,
                    )
                coords.append(
                    np.append(
                        mask_contours["data"]["coord_list"][0],
                        [mask_contours["data"]["coord_list"][0][0]],
                        axis=0
                        )
                    )
            if flag_verbose:
                print("Found {} circles".format(len(circles[0])))
        else:
            if flag_verbose:
                print("No circles detected")
            return None
        
        
	# =============================================================================
	# assemble results
    
    annotation = {
        "info": {
            "annotation_type": "mask",
            "pp_function": "mask_detect",
        },
        "settings": local_settings,
        "data": {
            "include": include,
            "n_masks": len(coords),
            "coord_list": coords,
        }
    }
    
	# =============================================================================
	# return
    
    return annotation
        
    

def create_reference(
    image,
    mask=False,
    **kwargs
):
    """
    Measure a size or colour reference card. Minimum input interaction is 
    measuring a size reference: click on two points inside the provided image, 
    and enter the distance - returns the pixel-to-mm-ratio. 
    
    In an optional second step, drag a rectangle mask over the reference card 
    to exclude it from anysubsequent image segementation. The mask can also be 
    stored as a template for automatic reference detection with the
    "detect_reference" function.

    Parameters
    ----------
    image: ndarray 
        input image
    mask : bool, optional
        mask a reference card inside the image and return its coordinates as 

    Returns
    -------
    annotation_ref: dict
        phenopype annotation containing mask coordinates pixel to mm ratio
    annotation_mask: dict
        phenopype annotation containing mask coordinates

    """



    # =============================================================================
    # setup 
    
    flags = make_dataclass(cls_name="flags", 
                           fields=[("mask", bool, mask)])   
    # =============================================================================
    # retain settings

    
    # =============================================================================
    # exectue

    out = _ImageViewer(image, tool="reference")
    
    ## enter length
    points = out.reference_coords
    distance_px = _sqrt(
            ((points[0][0] - points[1][0]) ** 2)
            + ((points[0][1] - points[1][1]) ** 2)
        )
    
    out = _ImageViewer(image, tool="comment", display="Enter distance in mm:")
    entry = out.entry
    distance_mm = float(entry)
    px_mm_ratio = float(distance_px / distance_mm)
    
	# =============================================================================
	# assemble results

    annotation_ref = {
        "info": {
            "annotation_type": "reference",
            "pp_function": "create_reference",
        },
        "data": {
            "px_mm_ratio":px_mm_ratio
        }
    }
    
    ## mask reference
    if flags.mask:
        out = _ImageViewer(image, tool="template")
        
        annotation_mask = {
            "info": {
                "annotation_type": "mask",
                "pp_function": "create_reference",
            },
            "data": {
                "include": False,
                "n_masks": 1,
                "coord_list": out.polygons   
            }
        }
        
        return annotation_ref, annotation_mask
    else:
        return annotation_ref



def detect_reference(
    image,
    image_template,
    px_mm_ratio_template,
    mask=True,
    equalize=False,
    min_matches=10,
    resize=1,
    **kwargs,
):
    """
    Find reference from a template created with "create_reference". Image registration 
    is run by the "AKAZE" algorithm. Future implementations will include more 
    algorithms to select from. First, use "create_reference" with "mask=True"
    and pass the template to this function. This happends automatically in the 
    low and high throughput workflow. Use "equalize=True" to adjust the 
    histograms of all colour channels to the reference image.
    
    AKAZE: http://www.bmva.org/bmvc/2013/Papers/paper0013/abstract0013.pdf

    Parameters
    -----------
    image: ndarray 
        input image
    image_template : array
        reference image of reference
    equalize : bool, optional
        should the provided image be colour corrected to match the template 
        images' histogram
    min_matches : int, optional
       minimum key point matches for image registration
    resize: num, optional
        resize image to speed up detection process. default: 0.5 for 
        images with diameter > 5000px (WARNING: too low values may 
        result in poor detection performance or even crashes)
    px_mm_ratio_template : int, optional
        pixel-to-mm-ratio of the template image

    Returns
    -------
    annotation_ref: dict
        phenopype annotation containing mask coordinates pixel to mm ratio
    annotation_mask: dict
        phenopype annotation containing mask coordinates
    """

    ## kwargs
    flags = make_dataclass(cls_name="flags", 
                           fields=[("mask", bool, mask),
                                   ("equalize",bool, equalize)])   
         
    
    # =============================================================================
    # retain settings


    ## if image diameter bigger than 5000 px, then automatically resize
    if (image.shape[0] + image.shape[1]) / 2 > 5000 and resize == 1:
        resize_factor = 0.5
        print(
            "large image - resizing by factor "
            + str(resize_factor)
            + " to avoid slow image registration"
        )
    else:
        resize_factor = resize
    image = cv2.resize(image, (0, 0), fx=1 * resize_factor, fy=1 * resize_factor)

    # =============================================================================
    # exectue function
    
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(image_template, None)
    kp2, des2 = akaze.detectAndCompute(image, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.knnMatch(des1, des2, 2)

    # keep only good matches
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # find and transpose coordinates of matches
    if len(good) >= min_matches:
        
        ## find homography betweeen detected keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        ## transform boundary box of template
        rect_old = np.array(
            [
                [[0, 0]],
                [[0, image_template.shape[0]]],
                [[image_template.shape[1], image_template.shape[0]]],
                [[image_template.shape[1], 0]],
            ],
            dtype=np.float32,
        )
        rect_new = cv2.perspectiveTransform(rect_old, M) / resize_factor

        # calculate template diameter
        (x, y), radius = cv2.minEnclosingCircle(rect_new.astype(np.int32))
        diameter_new = radius * 2

        # calculate transformed diameter
        (x, y), radius = cv2.minEnclosingCircle(rect_old.astype(np.int32))
        diameter_old = radius * 2

        ## calculate ratios
        diameter_ratio = diameter_new / diameter_old
        px_mm_ratio_detected = round(diameter_ratio * px_mm_ratio_template, 1)

        ## feedback
        print("---------------------------------------------------")
        print("Reference card found with %d keypoint matches:" % len(good))
        print("template image has %s pixel per mm." % (px_mm_ratio_template))
        print("current image has %s pixel per mm." % (px_mm_ratio_detected))
        print("= %s %% of template image." % round(diameter_ratio * 100, 3))
        print("---------------------------------------------------")

        ## create mask from new coordinates
        rect_new = rect_new.astype(int)
        coords_list = _convert_arr_tup_list(rect_new)
        coords_list[0].append(coords_list[0][0])
        
    else:
        ## feedback
        print("---------------------------------------------------")
        print("Reference card not found - %d keypoint matches:" % len(good))
        print('Setting "current reference" to None')
        print("---------------------------------------------------")
        px_mm_ratio_detected = None
        

    ## do histogram equalization
    if flags.equalize:
        detected_rect_mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(detected_rect_mask, [np.array(rect_new)], colours["white"])
        (rx, ry, rw, rh) = cv2.boundingRect(np.array(rect_new))
        detected_rect_mask = ma.array(
            data=image[ry : ry + rh, rx : rx + rw],
            mask=detected_rect_mask[ry : ry + rh, rx : rx + rw],
        )
        image = _equalize_histogram(image, detected_rect_mask, image_template)
        print("histograms equalized")

    annotation_ref = {
        "info": {
            "annotation_type": "reference",
            "pp_function": "detect_reference",
        },
        "data": {
            "px_mm_ratio":px_mm_ratio_detected
        }
    }
    

    ## mask reference
    if flags.mask:
        annotation_mask = {
            "info": {
                "annotation_type": "mask",
                "pp_function": "detect_reference",
            },
            "data": {
                "include": False,
                "n_masks": 1,
                "coord_list": coords_list
            }
        }
        
        return annotation_ref, annotation_mask
    else:
        return annotation_ref



def comment(
    image,
    field="ID",
    **kwargs
):
    """
    Add a comment. 

    Parameters
    ----------
    image : ndarray
        input image
    field : str, optional
        name the comment-field (useful for later processing). The default is "ID".

    Returns
    -------
    annotation_ref: dict
        phenopype annotation containing comment

    """

	# =============================================================================
	# setup 
    
    annotation_previous = kwargs.get("annotation_previous")
        
    # =============================================================================
    # retain settings

    ## retrieve settings from args
    local_settings  = _drop_dict_entries(locals(),
        drop=["image","kwargs","annotation_previous"])
    
    ## retrieve update IV settings and data from previous annotations  
    IV_settings = {}     
    if annotation_previous:    
        IV_settings["ImageViewer_previous"] =_load_previous_annotation(
            annotation_previous = annotation_previous, 
            components = [
                ("data","field"),
                ("data","entry")
                ])            
        
    ## update local and IV settings from kwargs
    if kwargs:
        _update_settings(kwargs, local_settings, IV_settings)
        
	# =============================================================================
	# execute
    
    out = _ImageViewer(image, 
                       tool="comment", 
                       field=field, 
                        **IV_settings)
    

	# =============================================================================
	# assemble results

    annotation = {
        "info": {
            "annotation_type": "comment",
            "pp_function": "comment",
        },
        "settings": local_settings,
        "data": {
            "field": out.field,
            "entry": out.entry
        }
    }
    
    
	# =============================================================================
	# return

    return annotation



def decompose_image(image, 
                    channel="gray", 
                    invert=False):
    
    """
    Extract single channel from multi-channel array.

    Parameters
    ----------
    image : ndarray
        input image
    channel : {"raw", "gray", "red", "green", "blue", "hue", "saturation", "value"} str, optional
        select specific image channel
    invert: false, bool
        invert all pixel intensities in image (e.g. 0 to 255 or 100 to 155)
        
    Returns
    -------
    image : ndarray
        decomposed image.

    """
    
	# =============================================================================
	# execute
    
    if len(image.shape) == 2: 
        print("- single channel image supplied - no decomposition possible")
        pass
    elif len(image.shape) == 3: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if channel in ["grayscale","gray"]:
            image = gray
        elif channel in ["green","g"]:
            image = image[:, :, 0]
        elif channel in ["red", "r"]:
            image = image[:, :, 1]
        elif channel in ["blue","b"]:
            image = image[:, :, 2]
        elif channel in ["hue","h"]:
            image = hsv[:, :, 0]
        elif channel in ["saturation", "sat", "s"]:
            image = hsv[:, :, 1]
        elif channel in ["value","v"]:
            image = hsv[:, :, 2]
        elif channel == "raw":
            pass
        else:
            print("- don't know how to handle channel {}".format(channel))
            return 
            
        if flag_verbose:
            print("- decompose image: using {} channel".format(str(channel)))
        
    if invert==True:
        image = cv2.bitwise_not(image)
        print("- inverted image")
        
        
	# =============================================================================
	# return
        
    return image


