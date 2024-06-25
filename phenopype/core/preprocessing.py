#%% modules

import copy
import cv2
import math
import numpy as np
import sys
import time
from dataclasses import make_dataclass
from tqdm import tqdm

from phenopype import __version__
from phenopype import _vars
from phenopype import config
from phenopype import decorators
from phenopype import utils
from phenopype import utils_lowlevel as ul
from phenopype import core


#%% functions

def blur(
    image, 
    kernel_size=5, 
    method="averaging", 
    sigma_color=75,
    sigma_space=75, 
    verbose=False, 
    custom_kernel=None,
    **kwargs
    ):
    """
    Apply a blurring algorithm to an image with enhanced features.

    Parameters
    ----------
    image : ndarray
        The input image.
    kernel_size : int, optional
        Size of the blurring kernel, must be positive and odd.
    method : str, optional
        Blurring algorithm: 'averaging', 'gaussian', 'median', 'bilateral', or 'custom'.
    sigma_color : int, optional
        For 'bilateral' filter, the filter sigma in the color space.
    sigma_space : int, optional
        For 'bilateral' filter, the filter sigma in the coordinate space.
    verbose : bool, optional
        If True, prints additional details about the process.
    custom_kernel : ndarray, optional
        Custom kernel for convolution if method is 'custom'.

    Returns
    -------
    ndarray
        The blurred image.
    """

    if kernel_size % 2 == 0:
        kernel_size += 1  # Make kernel_size odd if it is even
    
    if verbose:
        print(f"Applying {method} blur with kernel size {kernel_size}")

    if method == "averaging":
        blurred = cv2.blur(image, (kernel_size, kernel_size))
    elif method == "gaussian":
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        blurred = cv2.medianBlur(image, kernel_size)
    elif method == "bilateral":
        blurred = cv2.bilateralFilter(image, kernel_size, sigma_color, sigma_space)
    elif method == "custom":
        blurred = cv2.filter2D(image, -1, custom_kernel)

    return blurred


def clip_histogram(image, percent=1):
    """
    Enhances the contrast of an image by clipping the histogram and applying histogram stretching.

    Parameters
    ----------
    image : ndarray
        The input grayscale image.
    percent : float, optional
        The percentage of the histogram to be clipped from both tails. Default is 1%.

    Returns
    -------
    image : ndarray
        The contrast-enhanced image.

    """

    if not (0 <= percent <= 100):
        raise ValueError("Percent must be between 0 and 100")

    # Calculate the histogram of the image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    # Calculate the cumulative distribution of the histogram
    accumulator = np.cumsum(hist)

    # Calculate the total number of pixels to clip from each tail of the histogram
    clip_value = percent * 0.5 * accumulator[-1] / 100

    # Determine the grayscale thresholds to clip
    minimum_gray = np.searchsorted(accumulator, clip_value)
    maximum_gray = np.searchsorted(accumulator, accumulator[-1] - clip_value) - 1

    # Handle cases where the percent value is too high, causing minimum_gray to exceed maximum_gray
    if minimum_gray >= maximum_gray:
        raise ValueError("Clipping percent too high; all pixels fall within the clip range.")

    # Apply histogram stretching
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # Convert scale and apply the calculated alpha and beta
    contrast_enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return contrast_enhanced_image



def create_mask(
    image,
    tool="rectangle",
    include=True,
    label=None,
    line_colour="default",
    line_width="auto",
    label_colour="default",
    label_size="auto",
    label_width="auto",
    **kwargs,
):

    """    
    Mask an area by drawing a rectangle or polygon. Multiple mask components count
    as the same mask - e.g., if objects that you would like to mask out or include
    can be scattered across the image. Rectangles will finish upon lifting the mouse
    button, polygons are completed by pressing CTRL. 
    
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
        label string for this mask and all its components
    line_colour: {"default", ... see phenopype.print_colours()} str, optional
        contour line colour - default colour as specified in settings
    line_width: {"auto", ... int > 0} int, optional 
        contour line width - automatically scaled to image by default
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        contour label colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        contour label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        contour label font thickness - automatically scaled to image by default
    
    Returns
    -------
    annotations: dict
        phenopype annotation containing mask coordinates

    """

    # =============================================================================
    # annotation management

    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    gui_data = {_vars._coord_list_type: ul._get_GUI_data(annotation)}
    gui_settings = ul._get_GUI_settings(kwargs, annotation)

    # =============================================================================
    # execute function

    gui = ul._GUI(
        image=image,
        tool=tool,
        line_width=line_width,
        line_colour=line_colour,
        label=label,
        label_size=label_size,
        label_width=label_width,
        label_colour=label_colour,
        data=gui_data,
        **gui_settings,
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
            "tool": tool,
            "line_width": line_width,
            "line_colour": line_colour,
            "label_size": label_size,
            "label_width": label_width,
            "label_colour": label_colour,
        },
        "data": {
            "label": label,
            "include": include,
            "n": len(gui.data[_vars._coord_list_type]),
            annotation_type: gui.data[_vars._coord_list_type],
        },
    }

    if len(gui_settings) > 0:
        annotation["settings"]["GUI"] = gui_settings

    # =============================================================================
    # return

    return ul._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )


def detect_mask(
    image,
    include=True,
    label=None,
    shape="circle",
    resize=1,
    circle_args={
        "dp": 1,
        "min_dist": 50,
        "param1": 200,
        "param2": 100,
        "min_radius": 0,
        "max_radius": 0,
    },
    **kwargs,
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
        phenopype annotation

    """

    # =============================================================================
    # annotation management

    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
    
    # =============================================================================
    # setup

    if len(image.shape) == 3:
        image = decompose_image(image, "gray")
    image_resized = utils.resize_image(image, resize)

    circle_args_exec = {
        "dp": 1,
        "min_dist": 50,
        "param1": 200,
        "param2": 100,
        "min_radius": 0,
        "max_radius": 0,
    }

    circle_args_exec.update(circle_args)

    # =============================================================================
    # execute

    if shape == "circle":
        circles = cv2.HoughCircles(
            image_resized,
            cv2.HOUGH_GRADIENT,
            dp=max(int(circle_args_exec["dp"] * resize), 1),
            minDist=int(circle_args_exec["min_dist"] * resize),
            param1=int(circle_args_exec["param1"] * resize),
            param2=int(circle_args_exec["param2"] * resize),
            minRadius=int(circle_args_exec["min_radius"] * resize),
            maxRadius=int(circle_args_exec["max_radius"] * resize),
        )

        ## output conversion
        circle_masks= []
        if circles is not None:
            for idx, circle in enumerate(circles[0]):
                x, y, radius = circle / resize
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask = cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
                mask_contours = core.segmentation.detect_contour(
                    mask, retrieval="ext", approximation="KCOS", verbose=False,
                )
                mask_coords = mask_contours["contour"]["a"]["data"][_vars._contour_type][0]
                mask_coords = [np.append(
                        np.concatenate(mask_coords),
                        [np.concatenate(mask_coords[0])],
                        axis=0,
                    )]
                circle_masks.append(np.concatenate(mask_coords))
            print("Found {} circles".format(len(circles[0])))
        else:
            print("No circles detected")
            return None

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
            "tool": "polygon",
        },
        "data": {
            "label": label,
            "include": include,
            "n": len(circle_masks),
            annotation_type: circle_masks,
        },
    }

    # =============================================================================
    # return

    return ul._update_annotations(
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
    **kwargs,
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
    line_colour: {"default", ... see phenopype.print_colours()} str, optional
        contour line colour - default colour as specified in settings
    line_width: {"auto", ... int > 0} int, optional 
        contour line width - automatically scaled to image by default
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        contour label colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        contour label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        contour label font thickness - automatically scaled to image by default
        
    Returns
    -------
    annotations: dict
        phenopype annotation

    """
    # =============================================================================
    # setup

    label = kwargs.get("label")

    # =============================================================================
    # annotation management

    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    gui_settings = ul._get_GUI_settings(kwargs, annotation)

    ## not pretty but needed for tests:
    gui_data = {}
    if annotation:
        gui_data.update({_vars._coord_type: annotation["data"]["support"]})
        gui_data.update(
            {_vars._comment_type: annotation["data"][annotation_type][0]}
        )
        unit = annotation["data"][annotation_type][0]
        label = annotation["data"]["label"]
        if "mask" in annotation["data"]:
            gui_data.update(
                {_vars._coord_list_type: annotation["data"][_vars._mask_type]}
            )

    # =============================================================================
    # execute

    ## measure length
    gui = ul._GUI(
        image,
        tool="reference",
        line_width=line_width,
        line_colour=line_colour,
        data=gui_data,
        **gui_settings,
    )

    ## enter length
    points = gui.data[_vars._coord_type]
    distance_px = math.sqrt(
        ((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2)
    )

    ## enter distance
    gui = ul._GUI(
        image,
        tool="comment",
        query="distance in {}".format(unit),
        label_size=label_size,
        label_width=label_width,
        label_colour=label_colour,
        data=gui_data,
        **gui_settings,
    )

    ## output conversion
    distance_measured = float(gui.data[_vars._comment_type])
    px_ratio = round(float(distance_px / distance_measured), 3)

    if mask:
        gui = ul._GUI(
            image=image,
            tool="rectangle",
            line_width=line_width,
            line_colour=line_colour,
            data=gui_data,
            **gui_settings,
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
            "label": label,
            annotation_type: (px_ratio, unit),
            "support": points,
            _vars._mask_type: gui.data[_vars._coord_list_type],
        },
    }

    if len(gui_settings) > 0:
        annotation["settings"]["GUI"] = gui_settings

    # =============================================================================
    # return

    return ul._update_annotations(
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
    template_id="a",
    get_mask=True,
    manual_fallback=True,
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
        template image-crop containing reference card
    px_ratio:
        px-mm-ratio of template image
    get_mask: bool
        retrieve mask and create annotation. The default is True.
    manual_fallback: bool
        use manual reference-tool in case detection fails. The default is True.
    correct_colours : bool, optional
        should the provided image be colour corrected to match the template 
        images' histogram
    min_matches : int, optional
       minimum key point matches for image registration
    resize: num, optional
        resize image to speed up detection process. default: 0.5 for 
        images with diameter > 5000px (WARNING: too low values may 
        result in poor detection performance or even crashes)
  
    Returns
    -------
    annotations: dict
        phenopype annotation
    """

    # =============================================================================
    # annotation management

    fun_name = sys._getframe().f_code.co_name

    annotations = kwargs.get("annotations", {})
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    # =============================================================================
    # setup

    ## kwargs
    flags = make_dataclass(
        cls_name="flags",
        fields=[
            ("mask", bool, get_mask), 
            ("equalize", bool, correct_colours),
            ("success", bool, False),
            ("homo", bool, False),
            ],
    )

    px_ratio_template = float(px_ratio)

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
    image_resized = cv2.resize(
        image, (0, 0), fx=1 * resize_factor, fy=1 * resize_factor
    )

    # =============================================================================
    # execute function

    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(template, None)
    kp2, des2 = akaze.detectAndCompute(image_resized, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

    good,coord_list,px_ratio_detected = [], [], None
    
    while True:
        if des2.__class__.__name__ == "NoneType":
            break
        
        # keep only good matches
        matches = matcher.knnMatch(des1, des2, 2)
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
                
        # find and transpose coordinates of matches
        if len(good) < min_matches:
            break

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

        if M.__class__.__name__ == "NoneType":
            break
        else:
            flags.homo = True          

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
        
        ## end
        flags.success = True
        break
        
        
    if flags.success:


        ## feedback
        print("---------------------------------------------------")
        print("Reference card found with {} keypoint matches:".format(len(good)))
        print(
            "template image has {} pixel per {}.".format(
                round(px_ratio_template, 3), unit
            )
        )
        print(
            "current image has {} pixel per mm.".format(px_ratio_detected)
        )
        print("= {} %% of template image.".format(round(diameter_ratio * 100, 3)))
        print("---------------------------------------------------")
    
        ## create mask from new coordinates
        rect_new = rect_new.astype(int)
        coord_list = ul._convert_arr_tup_list(rect_new)
        coord_list[0].append(coord_list[0][0])
        
        ## do histogram equalization
        if flags.equalize:
            detected_rect_mask = np.zeros(image.shape, np.uint8)
            cv2.fillPoly(
                detected_rect_mask, [np.array(rect_new)], ul._get_bgr("white")
            )
            (rx, ry, rw, rh) = cv2.boundingRect(np.array(rect_new))
            detected_rect_mask = np.ma.array(
                data=image[ry : ry + rh, rx : rx + rw],
                mask=detected_rect_mask[ry : ry + rh, rx : rx + rw],
            )
            image = ul._equalize_histogram(image, detected_rect_mask, template)
            print("histograms equalized")
    

    else:

        ## feedback
        print("---------------------------------------------------")
        print("Reference card not found - %d keypoint matches:" % len(good))
        if not flags.homo:
            print("No homography found!")
        print('Setting "current reference" to None')
        print("---------------------------------------------------")
        return



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
        "data": {annotation_type: (px_ratio_detected, unit),},
    }

    if flags.success and flags.mask:
        annotation["data"][_vars._mask_type] = coord_list

    # =============================================================================
    # return

    return ul._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

@decorators.annotation_function
def detect_QRcode(
    image,
    rot_steps=10,
    preprocess=True,
    rotate=True,
    max_dim=1000,
    enter_manually=False,
    label="ID",
    label_colour="default",
    label_size="auto",
    label_width="auto",
    **kwargs,
):
    """
    Find and read out a QR code that is contained inside an image. Rotate image
    until code is detected, or enter code manually if detection fails.

    Parameters
    ----------
    image: ndarray 
        input image
    rot_steps : int, optional
        angle by which image is rotated (until 360 is reached). The default is 20.
    enter_manually : bool, optional
        enter code manually if detection fails. The default is False.
    show_results : bool, optional
        show the detection results. The default is False.
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        text colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        text label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        text label font thickness - automatically scaled to image by default

    Returns
    -------
    annotations: dict
        phenopype annotation
    """
    
    # =============================================================================
    # annotation management

    fun_name = sys._getframe().f_code.co_name
    annotation_type = ul._get_annotation_type(fun_name)

    annotation = kwargs.get("annotation")
        
    gui_data = {_vars._comment_type: ul._get_GUI_data(annotation)}
    gui_settings = ul._get_GUI_settings(kwargs, annotation)
    
    # =============================================================================
    # setup
    
    flags = make_dataclass(
        cls_name="flags",
        fields=[("found", bool, False), 
                ("enter_manually", bool, enter_manually),
                ("preprocess", bool, preprocess),
                ("rotate", bool, rotate),
                ],
    )    

    # =============================================================================
    # execute
    
    image_copy, resize_factor = utils.resize_image(image.copy(), max_dim=max_dim, factor_ret=True)
    
    
    # Initialize QR-code detector
    qrCodeDetector = cv2.QRCodeDetector()
    
    # Attempt to decode QR code from the original image
    decodedText, points = qrCodeDetector.detectAndDecode(image_copy)[:2]
    
    # If not found, apply preprocessing and attempt again
    if points is None or points.size == 0 or decodedText=="":
        ul._print("- preprocessing image", lvl=1)
        image_prep = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        image_prep = cv2.GaussianBlur(image_prep, (5, 5), 0)
        _, image_prep = cv2.threshold(image_prep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        decodedText, points = qrCodeDetector.detectAndDecode(image_prep)[:2]
    
    # Rotate image and attempt decoding for both unprocessed and preprocessed images
    if points is None or points.size == 0 or decodedText=="":
        ul._print("- rotating image", lvl=1)
        for angle in range(0, 360, rot_steps):
            for image_variant in [image_copy, image_prep]:
                image_rot, image_center = ul._rotate_image(image_variant, angle, allow_crop=True, ret=True)
                decodedText, points_rot = qrCodeDetector.detectAndDecode(image_rot)[:2]
                if decodedText and points_rot is not None:
                    flags.found = True
                    points = ul._rotate_coords_center(points_rot, image_center, angle)
                    break
            if flags.found:
                break
    else:
        flags.found = True
         
    # Format points
    if flags.found:
        points = (points / resize_factor).astype(int)
        points = ul._convert_arr_tup_list(points)
        ul._print("- found QRcode: '{}'".format(decodedText))
    else:
        if flags.enter_manually:
            ul._print("- did not find QR-code - enter manually:")
            gui = ul._GUI(
                image,
                tool="comment",
                label="code",
                label_size=label_size,
                label_width=label_width,
                label_colour=label_colour,
                data=gui_data,
                **gui_settings,
            )
            decodedText = gui.data[_vars._comment_type]
        else:
            ul._print("- did not find QR-code")
            return
                
                
    # =============================================================================
    # assemble results

    annotation = {
        "info": {
            "annotation_type": annotation_type,
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
        },
        "settings": {
            "rotation_steps": rot_steps,
            "label_size": label_size,
            "label_width": label_width,
            "label_colour": label_colour,
        },
        "data": {
            annotation_type: decodedText,
            "label": label,
            _vars._mask_type: points,
            },
    }

    if len(gui_settings) > 0:
        annotation["settings"]["GUI"] = gui_settings

    # =============================================================================
    # return
    
    return annotation


def decompose_image(
    image, channel="gray", invert=False, 
    **kwargs
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
    # execute

    if len(image.shape) == 2:
        ul._print("- single channel image supplied - no decomposition possible", lvl=2)
        pass
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if channel in ["grayscale", "gray"]:
            image = gray
        elif channel in ["blue", "b"]:
            image = image[:, :, 0]
        elif channel in ["green", "g"]:
            image = image[:, :, 1]
        elif channel in ["red", "r"]:
            image = image[:, :, 2]
        elif channel in ["hue", "h"]:
            image = hsv[:, :, 0]
        elif channel in ["saturation", "sat", "s"]:
            image = hsv[:, :, 1]
        elif channel in ["value", "v"]:
            image = hsv[:, :, 2]
        elif channel == "raw":
            pass
        else:
            ul._print("- don't know how to handle channel {}".format(channel), lvl=1)
            return

        ul._print("- decompose image: using {} channel".format(str(channel)))

    if invert == True:
        image = cv2.bitwise_not(image)
        ul._print("- inverted image")

    # =============================================================================
    # return

    return image



def write_comment(
    image,
    label="ID",
    label_colour="default",
    label_size="auto",
    label_width="auto",
    **kwargs,
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
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    gui_settings = ul._get_GUI_settings(kwargs, annotation)
    gui_data = {_vars._comment_type: ul._get_GUI_data(annotation)}
    if annotation:
        label = annotation["data"]["label"]
        
    # =============================================================================
    # execute

    gui = ul._GUI(
        image,
        tool="comment",
        query=label,
        label_size=label_size,
        label_width=label_width,
        label_colour=label_colour,
        data=gui_data,
        **gui_settings,
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
            annotation_type: gui.data[_vars._comment_type],
            },
    }

    if len(gui_settings) > 0:
        annotation["settings"]["GUI"] = gui_settings

    # =============================================================================
    # return

    return ul._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
