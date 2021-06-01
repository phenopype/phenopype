#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from math import inf

from phenopype.settings import colours, flag_verbose, _image_viewer_arg_list
from phenopype.utils import image_select_channel, image_invert
from phenopype.utils_lowlevel import _create_mask_bool, _image_viewer, _auto_line_width

import phenopype.core.preprocessing as preprocessing
import phenopype.core.visualization as visualization


#%% functions



# pp.show_image(image)

def contour_detect(
    image,
    approximation="simple",
    retrieval="ext",
    offset_coords=[0,0],       
    min_nodes=3,
    max_nodes=inf,
    min_area=0,
    max_area=inf,
    min_diameter=0,
    max_diameter=inf,
    **kwargs,
    ):
        
    """
    Find objects in binarized image. The input image needs to be a binarized image
    or a container on which a binarizing segmentation algorithm (e.g. threshold or 
    watershed) has been performed. A flavor of different approximation algorithms 
    and retrieval intstructions can be applied. The contours can also be filtered
    for minimum area, diameter or nodes (= number of corners). 

    Parameters
    ----------
    image : array type
        input image (binary)
    approximation : {"none", "simple", "L1", "KCOS"] str, optional
        contour approximation algorithm:
            - none: no approximation, all contour coordinates are returned
            - simple: only the minimum coordinates required are returned
            - L1: Teh-Chin chain approximation algorithm 
            - KCOS: Teh-Chin chain approximation algorithm 
    retrieval : {"ext", "list", "tree", "ccomp", "flood"} str, optional
        contour retrieval procedure:
            - ext: retrieves only the extreme outer contours
            - list: retrieves all of the contours without establishing any 
              hierarchical relationships
            - tree: retrieves all of the contours and reconstructs a full 
              hierarchy of nested contours
            - ccomp: retrieves all of the contours and organizes them into a 
              two-level hierarchy (parent and child)
            - flood: flood-fill algorithm
    offset_coords : tuple, optional
        offset by which every contour point is shifted.
    subset: {"parent", "child"} str, optional
        retain only a subset of the parent-child order structure
    min_nodes : int, optional
        minimum number of coordinates
    max_nodes : int, optional
        maximum number of coordinates 
    min_area : int, optional
        minimum contour area in pixels
    max_area : int, optional
        maximum contour area in pixels
    min_diameter : int, optional
        minimum diameter of boundary circle
    max_diameter : int, optional
        maximum diameter of boundary circle

    Returns
    -------
    df_contours : DataFrame or container
        contains contours

    """    
    
    
    ## check
    if len(image.shape) > 2:
        print("Multi-channel array supplied - need binary array.")
        return 
    
    ## settings
    settings = locals()
    for rm in ["image"]:
        settings.pop(rm, None)
        
    ## definitions
    opencv_contour_flags = {
        "retrieval" : {
            "ext": cv2.RETR_EXTERNAL,  ## only external
            "list": cv2.RETR_LIST,  ## all contours
            "tree": cv2.RETR_TREE,  ## fully hierarchy
            "ccomp": cv2.RETR_CCOMP,  ## outer perimeter and holes
            "flood": cv2.RETR_FLOODFILL, ## not sure what this does
            },
        "approximation" : {
            "none": cv2.CHAIN_APPROX_NONE,  ## all points (no approx)
            "simple": cv2.CHAIN_APPROX_SIMPLE,  ## minimal corners
            "L1": cv2.CHAIN_APPROX_TC89_L1,  
            "KCOS": cv2.CHAIN_APPROX_TC89_KCOS, 
            }
        }

    ## method
    image, contours, hierarchies = cv2.findContours(
        image=image,
        mode=opencv_contour_flags["retrieval"][retrieval],
        method=opencv_contour_flags["approximation"][approximation],
        offset=tuple(offset_coords),
    )
   
    ## output conversion
    if contours is not None:
        coords, support = [], []
        for idx, (contour, hierarchy) in enumerate(zip(contours, hierarchies[0])):
            
            ## number of contour nodes
            if len(contour) > min_nodes and len(contour) < max_nodes:
            
                ## measure basic shape features
                center, radius = cv2.minEnclosingCircle(contour)
                center = [int(center[0]), int(center[1])]
                diameter = int(radius * 2)
                area = int(cv2.contourArea(contour))
    
                ## contour hierarchy
                if hierarchy[3] == -1:
                    hierarchy_level = "parent"
                else:
                    hierarchy_level = "child"
                    
                ## contour filter
                if all(
                        [
                            diameter > min_diameter and diameter < max_diameter,
                            area > min_area and area < max_area,
                            ]
                        ):
                
                    ## populate data lists
                    coords.append(contour)
                    support.append(
                        {
                            "center": center,
                            "area": area,
                            "diameter": diameter,
                            "hierarchy_level": hierarchy_level,
                            "hierarchy_idx_child": int(hierarchy[2]),
                            "hierarchy_idx_parent": int(hierarchy[3]),
                            }
                        )

        if flag_verbose:
            print("- found " + str(len(coords)) + " contours that match criteria")
    else:
        if flag_verbose:
            print("- no contours found")

    ## return
    ret = {
    "info":{
        "type": "contour", 
        "function": "contour_detect",
        "settings": settings
        },
    "data":{
        "coords": coords,
        "support": support,
        }
    }

    return ret


def contour_modify(
    image,
    contours,
    **kwargs
):
    """
    Set points, draw a connected line between them, and measure its length. 

    Parameters
    ----------
    obj_input : array or container
        input object
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata, will be added to
        output DataFrame

    Returns
    -------
    df_polylines : DataFrame or container
        contains the drawn polylines

    """
        
    ## retrieve settings for image viewer
    _image_viewer_settings = {}
    for key, value in kwargs.items():
        if key in _image_viewer_arg_list:
            _image_viewer_settings[key] = value

    ## settings
    settings = locals()
    for rm in ["image","contours","key","value",
               "_image_viewer_settings"]:
        settings.pop(rm, None)

    ## create binary overlay
    image_bin = np.zeros(image.shape, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(
            image=image_bin,
            contours=[contour],
            contourIdx=0,
            thickness=-1,
            color=colours["white"],
            maxLevel=3,
            offset=(0,0),
            )

    ## draw masks
    out = _image_viewer(image=image, 
                        image_bin=image_bin,
                        tool="draw", 
                        **_image_viewer_settings)
    if not out.done:
        print("- didn't finish: redo drawing")
        return 
    else:
        point_list = out.point_list
               
    ## draw
    for segment in point_list:
        cv2.polylines(
            image_bin,
            np.array([segment[0]]),
            False,
            segment[1],
            segment[2],
            )

    ## return
    if len(point_list) > 0:
        ret = {
            "info": {
                "type": "drawing", 
                "function": "contour_modify",
                "settings": settings,
                },
            "data":{
                "point_list": point_list,
                }
            }
        return ret
    else:
        print("- zero coordinates: redo drawing")
        return 



def morphology(obj_input, kernel_size=5, shape="rect", operation="close", iterations=1):
    """
    Performs advanced morphological transformations using erosion and dilation 
    as basic operations. Provides different kernel shapes and a suite of operation
    types (read more about morphological operations here:
    https://docs.opencv.org/master/db/df6/tutorial_erosion_dilatation.html)

    Parameters
    ----------
    obj_input : array or container
        input object
    kernel_size: int, optional
        size of the morphology kernel (has to be odd - even numbers will be 
        ceiled)
    shape : {"rect", "cross", "ellipse"} str, optional
        shape of the kernel
    operation : {erode, dilate, open, close, gradient, tophat, blackhat, hitmiss} str, optional
        the morphology operation to be performed:
            - erode: remove pixels from the border 
            - dilate: add pixels to the border
            - open: erosion followed by dilation
            - close: dilation followed by erosion
            - gradient: difference between dilation and erosion of an input image
            - tophat: difference between input image and opening of input image
            - blackhat: difference between input image and closing of input image
            - hitmiss: find patterns in binary images (read more here:
              https://docs.opencv.org/master/db/d06/tutorial_hitOrMiss.html)
    iterations : int, optional
        number of times to run morphology operation

    Returns
    -------
    image : array or container
        processed image

    """
    ## kwargs
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    shape_list = {
        "cross": cv2.MORPH_CROSS,
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    operation_list = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat ": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
        "hitmiss": cv2.MORPH_HITMISS,
    }
    kernel = cv2.getStructuringElement(shape_list[shape], (kernel_size, kernel_size))

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image

    ## method
    image = cv2.morphologyEx(
        image, op=operation_list[operation], kernel=kernel, iterations=iterations
    )

    ## return
    if obj_input.__class__.__name__ == "container":
        obj_input.image = image
        obj_input.image_bin = image
    else:
        return image



def threshold(
    image,
    method="otsu",
    constant=1,
    blocksize=99,
    value=127,
    channel="gray",
    invert=False,
    masks=None,
):
    """
    Applies a fixed-level threshold to create a binary image from a grayscale 
    image or a multichannel image (gray, red, green or blue channel can be selected).
    Three types of thresholding algorithms are supported. Binary mask coordinates
    can be supplied to include or exclude areas.

    Parameters
    ----------
    obj_input : array or container
        input object
    df_masks : DataFrame, optional
        contains mask coordinates
    method : {"otsu", "adaptive", "binary"} str, optional
        type of thresholding algorithm:
            - otsu: use Otsu algorithm to choose the optimal threshold value
            - adaptive: dynamic threshold values across image (uses arguments
              "blocksize" and "constant")
            - binary: fixed threshold value (uses argument "value")
    blocksize: int, optional
        Size of a pixel neighborhood that is used to calculate a threshold 
        value for the pixel (has to be odd - even numbers will be ceiled; for
        "adaptive" method)
    constant : int, optional
        value to subtract from binarization output (for "adaptive" method)
    value : {between 0 and 255} int, optional
        thesholding value (for "binary" method)
    channel {"gray", "red", "green", "blue"}: str, optional
        which channel of the image to use for thresholding 
    invert : bool, optional
        invert all pixel values PRIOR to applying threshold values

    Returns
    -------
    image : array or container
        binary image

    """


    ## checks
    if len(image.shape) == 3:
        image = image_select_channel(image)
        if flag_verbose:
            print("multi-channel supplied - defaulting to gray channel")
    if blocksize % 2 == 0:
        blocksize = blocksize + 1
        if flag_verbose:
            print("even blocksize supplied - need odd blocksize")
    if value > 255:
        value = 255
        if flag_verbose:
            print("warning - \"value\" has to be < 255")
    elif value < 0:
        value = 0
        if flag_verbose:
            print("warning - \"value\" has to be > 0")

    ## modifications
    if invert:
        image = image_invert(image)

    ## method
    if method == "otsu":
        ret, thresh = cv2.threshold(
            image, 
            0, 
            255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
    elif method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blocksize,
            constant,
        )
    elif method == "binary":
        ret, thresh = cv2.threshold(
            image, 
            value, 
            255, 
            cv2.THRESH_BINARY_INV
            )

    ## apply masks
    if not masks.__class__.__name__ == "NoneType":
        ## include == True
        mask_bool, include_idx, exclude_idx = np.zeros(image.shape, dtype=bool), 0,0
        for index, row in df_masks.iterrows():
            if row["include"] == True:
                if not row["mask"] == "":
                    coords = eval(row["coords"])
                    mask_bool = np.logical_or(mask_bool, _create_mask_bool(image, coords))
                    include_idx += 1
        if include_idx > 0:
            image[mask_bool == False] = 0
        for index, row in df_masks.iterrows():
            if row["include"] == False:
                if not row["mask"] == "":
                    coords = eval(row["coords"])
                    image[_create_mask_bool(image, coords)] = 0
                    exclude_idx += 1
        if exclude_idx>0:
            print("- excluding pixels from " + str(exclude_idx) + " drawn masks ")
        if include_idx>0:
            print("- including pixels from " + str(include_idx) + " drawn masks ")

    ## return
    return thresh



def watershed(
    obj_input,
    image_thresh=None,
    iterations=1,
    kernel_size=3,
    distance_cutoff=0.8,
    distance_mask=0,
    distance_type="l1",
):
    """
    Performs non-parametric marker-based segmentation - useful if many detected 
    contours are touching or overlapping with each other. Input image should be 
    a binary image that serves as the true background. Iteratively, edges are 
    eroded, the difference serves as markers.

    Parameters
    ----------
    obj_input : array or container
        input object
    image_thresh: array, optional
        thresholded image n
    kernel_size: int, optional
        size of the diff-kernel (has to be odd - even numbers will be ceiled)
    iterations : int, optional
        number of times to apply diff-operation
    distance_cutoff : {between 0 and 1} float, optional
        watershed distance transform cutoff (larger values discard more pixels)
    distance_mask : {0, 3, 5} int, optional
        size of distance mask - not all sizes work with all distance types (will
        be coerced to 0)
    distance_type : {"user", "l1", "l2", "C", "l12", "fair", "welsch", "huber"} str, optional
        distance transformation type

    Returns
    -------
    image : array or container
        binary image

    """

    ##kwargs
    distance_type_list = {
        "user": cv2.DIST_USER,
        "l1": cv2.DIST_L1,
        "l2": cv2.DIST_L2,
        "C": cv2.DIST_C,
        "l12": cv2.DIST_L12,
        "fair": cv2.DIST_FAIR,
        "welsch": cv2.DIST_WELSCH,
        "huber": cv2.DIST_HUBER,
    }
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
        thresh = copy.deepcopy(image_thresh)
    elif obj_input.__class__.__name__ == "container":
        thresh = copy.deepcopy(obj_input.image)
        image = copy.deepcopy(obj_input.image_copy)

    if thresh.__class__.__name__ == "NoneType":
        print("No thresholded version of image provided for watershed - aborting.")
        return
    if len(thresh.shape) == 3:
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    ## sure background
    ## note: sure_bg is set as the thresholded input image
    sure_bg= morphology(
        thresh,
        operation="dilate",
        shape="ellipse",
        kernel_size=kernel_size,
        iterations=iterations,
    )
    
    ## sure foreground
    if distance_type in ["user", "l12", "fair", "welsch", "huber"]:
        distance_mask = 0
    opened = morphology(
        thresh,
        operation="erode",
        shape="ellipse",
        kernel_size=kernel_size,
        iterations=iterations,
    )
    dist_transform = cv2.distanceTransform(
        opened, distance_type_list[distance_type], distance_mask
    )

    dist_transform = cv2.normalize(
        dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX
    )
    
    dist_transform = preprocessing.blur(dist_transform, kernel_size=int(2*kernel_size))
    dist_transform = abs(255 * (1-dist_transform)) 
    dist_transform = dist_transform.astype(np.uint8)

    sure_fg = threshold(dist_transform, method="binary", value=int(distance_cutoff*255))

    ## finding unknown region
    sure_fg = sure_fg.astype("uint8")
    sure_fg[sure_fg == 1] = 255
    unknown = cv2.subtract(sure_bg, sure_fg)

    ## marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ## watershed
    markers = cv2.watershed(preprocessing.blur(image, int(2*kernel_size)), markers)

    ## convert to contours
    watershed_mask = np.zeros(image.shape[:2], np.uint8)
    watershed_mask[markers == -1] = 255
    watershed_mask[0 : watershed_mask.shape[0], 0] = 0
    watershed_mask[0 : watershed_mask.shape[0], watershed_mask.shape[1] - 1] = 0
    watershed_mask[0, 0 : watershed_mask.shape[1]] = 0
    watershed_mask[watershed_mask.shape[0] - 1, 0 : watershed_mask.shape[1]] = 0

    contours = find_contours(watershed_mask, retrieval="ccomp")
    image_watershed = np.zeros(watershed_mask.shape, np.uint8)
    
    for index, row in contours.iterrows():
        if row["order"] == "child":
            cv2.drawContours(
                image=image_watershed,
                contours=[row["coords"]],
                contourIdx=0,
                thickness=-1,
                color=colours["white"],
                maxLevel=3,
                offset=None
                )
            cv2.drawContours(
                image=image_watershed,
                contours=[row["coords"]],
                contourIdx=0,
                thickness=2,
                color=colours["black"],
                maxLevel=3,
                offset=None
                )
    
    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image_watershed
    elif obj_input.__class__.__name__ == "container":
        obj_input.image = image_watershed
        obj_input.image_bin = image_watershed
