#%% modules

import cv2
import numpy as np
import sys

from dataclasses import make_dataclass
import math

from phenopype import __version__
from phenopype import settings
from phenopype import utils_lowlevel
from phenopype.core import segmentation

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
        if settings.flag_verbose:
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
        tool="rectangle",
        include=True,
        line_colour="default",
        line_width="auto",
        label_colour="default",
        label_size="auto",
        label_width="auto",
        **kwargs,
    ):
           
    """    
    Mask an area.
    
    ANNOTATION FUNCTION    

    Parameters
    ----------
    image: array 
        input image
    tool : {"rectangle","polygon"} str, optional
        Type of mask tool to be used. The default is "rectangle".
    include : bool, optional
        include or exclude area inside mask
    label : str, optional
        text label for this mask and all its components
        
    Returns
    -------
    annotations: dict
        phenopype annotation containing mask coordinates

    """
    
    # =============================================================================
    # annotation management
    
    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations, 
        annotation_type=annotation_type, 
        annotation_id=annotation_id, 
        kwargs=kwargs,
    )
            
    gui_data = {settings._coord_list_type: utils_lowlevel._get_GUI_data(annotation)}
    gui_settings = utils_lowlevel._get_GUI_settings(kwargs, annotation)
    
    # =============================================================================
    # setup
    
    label = kwargs.get("label")
    
    if line_width == "auto":
        line_width = utils_lowlevel._auto_line_width(image)
    if label_size == "auto":
        label_size = utils_lowlevel._auto_text_size(image)
    if label_width == "auto":
        label_width = utils_lowlevel._auto_text_width(image)
        
    if line_colour == "default":
        line_colour = settings._default_line_colour
    if label_colour == "default":
        label_colour = settings._default_label_colour
        
    label_colour = utils_lowlevel._get_bgr(label_colour)     
    line_colour = utils_lowlevel._get_bgr(line_colour)    
    
    # =============================================================================
	# execute function
        
    gui = utils_lowlevel._GUI(
        image=image, 
        tool=tool, 
        line_width=line_width,
        line_colour=line_colour,
        label=label,
        label_size=label_size,
        label_width=label_width,
        label_colour=label_colour,
        data=gui_data,
        **gui_settings
        )
            
	# =============================================================================
	# assemble results
        
    annotation = {
        "info": {
            "annotation_type": annotation_type,
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            },
        "settings": {
            "tool":tool,
            "line_width":line_width,
            "line_colour":line_colour,
            "label_size":label_size,
            "label_width":label_width,
            "label_colour":label_colour,
            },
        "data": {
            "label":label,
            "include":include,
            "n": len(gui.data[settings._coord_list_type]),
            annotation_type: gui.data[settings._coord_list_type],
            }
    }
    
    if len(gui_settings) > 0:
        annotation["settings"]["GUI"] = gui_settings
    
	# =============================================================================
	# return
    
    return utils_lowlevel._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    
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
            - min_dist: minimum distance between the centers of the detected circles
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
    # annotation management
    
    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotations_id", None)
    
    annotation = utils_lowlevel._get_annotation(
        annotations=annotations, 
        annotation_type=annotation_type, 
        annotation_id=annotation_id, 
        kwargs=kwargs,
    )

    # =============================================================================
    # setup
    
    label = kwargs.get("label")
    
    if len(image.shape) == 3:
        image = decompose_image(image, "gray")
    image_resized = utils_lowlevel._resize_image(image, resize)
    
    circle_args_exec = {
        "dp":1,
        "min_dist":50,
        "param1":200,
        "param2":100,
        "min_radius":0,
        "max_radius":0
        }
    
    circle_args_exec.update(circle_args)
    
    # =============================================================================
    # execute 
    
    if shape=="circle":
        circles = cv2.HoughCircles(
            image_resized, 
            cv2.HOUGH_GRADIENT, 
            dp=max(int(circle_args_exec["dp"]*resize),1), 
            minDist=int(circle_args_exec["min_dist"]*resize),
            param1=int(circle_args_exec["param1"]*resize), 
            param2=int(circle_args_exec["param2"]*resize),
            minRadius=int(circle_args_exec["min_radius"]*resize), 
            maxRadius=int(circle_args_exec["max_radius"]*resize)
            )
    
        ## output conversion
        if circles is not None:
            circle_contours = []
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
                circle_contours.append(
                    np.append(
                        mask_contours["contour"]["a"]["data"][settings._contour_type][0],
                        [mask_contours["contour"]["a"]["data"][settings._contour_type][0][0]],
                        axis=0
                        )
                    )
            print("Found {} circles".format(len(circles[0])))
        else:
            print("No circles detected")
            return
        
	# =============================================================================
	# assemble results
    
    annotation = {
        "info": {
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            "annotation_type": annotation_type,
            },
        "settings": {
            "shape": shape,
            "resize": resize,
            "circle_args": circle_args_exec,
            },
        "data": {
            "label":label,
            "include":include,
            "n": len(circle_contours),
            annotation_type: circle_contours,
            }
        }
    
	# =============================================================================
	# return
    
    return utils_lowlevel._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
    

def create_reference(
        image,
        unit="mm",
        line_colour="default",
        line_width="auto",
        label=True,
        label_colour="default",
        label_size="auto",
        label_width="auto",
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
    
    label = kwargs.get("label")
    
    # =============================================================================
    # annotation management
    
    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)
    
    annotation = utils_lowlevel._get_annotation(
        annotations=annotations, 
        annotation_type=annotation_type, 
        annotation_id=annotation_id, 
        kwargs=kwargs,
    )
    
    gui_settings = utils_lowlevel._get_GUI_settings(kwargs, annotation)
    
    ## not pretty but needed for tests:
    gui_data = {}
    if annotation:
        gui_data.update({settings._coord_type: annotation["data"]["support"]})
        gui_data.update({settings._comment_type: annotation["data"][annotation_type][0]})
        unit = annotation["data"][annotation_type][0]
        label = annotation["data"]["label"]
        if "mask" in annotation["data"]:
            gui_data.update({settings._coord_list_type: annotation["data"][settings._mask_type]})

    # =============================================================================
    # setup
        
    if line_width == "auto":
        line_width = utils_lowlevel._auto_line_width(image)
    if label_size == "auto":
        label_size = utils_lowlevel._auto_text_size(image)
    if label_width == "auto":
        label_width = utils_lowlevel._auto_text_width(image)
    if line_colour == "default":
        line_colour = settings._default_line_colour
    if label_colour == "default":
        label_colour = settings._default_label_colour
        
    label_colour = utils_lowlevel._get_bgr(label_colour)     
    line_colour = utils_lowlevel._get_bgr(line_colour)   
    
    # =============================================================================
    # execute

    ## measure length
    gui = utils_lowlevel._GUI(
        image, 
        tool="reference",
        line_width=line_width,
        line_colour=line_colour,
        data=gui_data,
        **gui_settings,
        )
        
    ## enter length
    points = gui.data[settings._coord_type]
    distance_px = math.sqrt(
            ((points[0][0] - points[1][0]) ** 2)
            + ((points[0][1] - points[1][1]) ** 2)
        )
    
    ## enter distance
    gui = utils_lowlevel._GUI(
        image, 
        tool="comment", 
        label="distance in {}".format(unit),
        label_size=label_size,
        label_width=label_width,
        label_colour=label_colour,
        data=gui_data,
        **gui_settings,
        )
    
    ## output conversion
    distance_measured = float(gui.data[settings._comment_type])
    px_ratio = round(float(distance_px / distance_measured), 3)

    if mask:
        gui = utils_lowlevel._GUI(
            image=image, 
            tool="rectangle", 
            line_width=line_width,
            line_colour=line_colour,
            data=gui_data,
            **gui_settings
            )
    
	# =============================================================================
	# assemble results

    annotation = {
        "info": {
            "annotation_type": annotation_type,
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
        },
        "settings": {},
        "data": {
            "label":label,
            annotation_type: (px_ratio, unit),
            "support": points,
            settings._mask_type: gui.data[settings._coord_list_type]
        }
    }
    
    if len(gui_settings) > 0:
        annotation["settings"]["GUI"] = gui_settings
    
	# =============================================================================
	# return
    
    return utils_lowlevel._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

def detect_reference(
        image,
        template,
        px_ratio,
        unit,
        get_mask=True,
        correct_colours=False,
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
    template : array
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

    # =============================================================================
    # annotation management
    
    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)
        
    # =============================================================================
    # setup 
    
    ## kwargs
    flags = make_dataclass(cls_name="flags", 
                           fields=[("mask", bool, get_mask),
                                   ("equalize",bool, correct_colours)])   
         
    px_ratio_template = px_ratio

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
    image_resized = cv2.resize(image, (0, 0), fx=1 * resize_factor, fy=1 * resize_factor)

    # =============================================================================
    # execute function
        
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(template, None)
    kp2, des2 = akaze.detectAndCompute(image_resized, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    
    good = []
    
    while True:
    
        if not des2.__class__.__name__ == "NoneType":
            matches = matcher.knnMatch(des1, des2, 2)
        else:
            break
    
        # keep only good matches
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
                    [[0, template.shape[0]]],
                    [[template.shape[1], template.shape[0]]],
                    [[template.shape[1], 0]],
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
            px_ratio_detected = round(diameter_ratio * px_ratio_template, 3)
    
            ## feedback
            print("---------------------------------------------------")
            print("Reference card found with {} keypoint matches:".format(len(good)))
            print("template image has {} pixel per {}.".format(round(px_ratio_template,3), unit))
            print("current image has {} pixel per mm.".format(round(px_ratio_detected,3)))
            print("= {} %% of template image.".format(round(diameter_ratio * 100, 3)))
            print("---------------------------------------------------")
    
            ## create mask from new coordinates
            rect_new = rect_new.astype(int)
            coord_list = utils_lowlevel._convert_arr_tup_list(rect_new)
            coord_list[0].append(coord_list[0][0])
            
            break
        
    if len(good) == 0:
        
        ## feedback
        print("---------------------------------------------------")
        print("Reference card not found - %d keypoint matches:" % len(good))
        print('Setting "current reference" to None')
        print("---------------------------------------------------")
        px_ratio_detected = None
        return
        

    ## do histogram equalization
    if flags.equalize:
        detected_rect_mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(detected_rect_mask, [np.array(rect_new)], utils_lowlevel._get_bgr("white"))
        (rx, ry, rw, rh) = cv2.boundingRect(np.array(rect_new))
        detected_rect_mask = np.ma.array(
            data=image[ry : ry + rh, rx : rx + rw],
            mask=detected_rect_mask[ry : ry + rh, rx : rx + rw],
        )
        image = utils_lowlevel._equalize_histogram(image, detected_rect_mask, template)
        print("histograms equalized")
        

    annotation = {
        "info": {
            "annotation_type": annotation_type,
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
        },
        "settings": {
            "get_mask": flags.mask,
            "correct_colours": flags.equalize,
            "min_matches": min_matches,
            "resize": resize_factor,
            },
        "data": {
            annotation_type: (px_ratio_detected, unit),
        }
    }  

    if flags.mask:
        annotation["data"][settings._mask_type] = coord_list
        

	# =============================================================================
	# return
    
    return utils_lowlevel._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )






def decompose_image(
        image, 
        channel="gray", 
        invert=False,
        **kwargs,
    ):
    
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
    ## setup 
        
    verbose = kwargs.get("verbose", settings.flag_verbose)
    
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
            
        if verbose:
            print("- decompose image: using {} channel".format(str(channel)))
        
    if invert==True:
        image = cv2.bitwise_not(image)
        print("- inverted image")
        
        
	# =============================================================================
	# return
        
    return image



def write_comment(
        image,
        label="ID",
        tool="rectangle",
        label_colour="default",
        label_size="auto",
        label_width="auto",
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
    # annotation management
    
    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations, 
        annotation_type=annotation_type, 
        annotation_id=annotation_id, 
        kwargs=kwargs,
    )
    
    gui_settings = utils_lowlevel._get_GUI_settings(kwargs, annotation)
    gui_data = {settings._comment_type: utils_lowlevel._get_GUI_data(annotation)}
    if annotation:
        label = annotation["data"]["label"]
        
    # =============================================================================
	# setup
    
    if label_size == "auto":
        label_size = utils_lowlevel._auto_text_size(image)
    if label_width == "auto":
        label_width = utils_lowlevel._auto_text_width(image)

    if label_colour == "default":
        label_colour = settings._default_label_colour
        
    label_colour = utils_lowlevel._get_bgr(label_colour)     
    

	# =============================================================================
	# execute
    
    print(label)
    
    gui = utils_lowlevel._GUI(
        image, 
        tool="comment", 
        label=label,
        label_size=label_size,
        label_width=label_width,
        label_colour=label_colour,
        data=gui_data,
         **gui_settings
         )
    

	# =============================================================================
	# assemble results

    annotation = {
        "info": {
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            "annotation_type": annotation_type,
        },
        "settings": {},
        "data": {
            "label": label,
            annotation_type: gui.data[settings._comment_type],
        }
    }
    
    if len(gui_settings) > 0:
        annotation["settings"]["GUI"] = gui_settings
        
	# =============================================================================
	# return
    
    return utils_lowlevel._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
    
