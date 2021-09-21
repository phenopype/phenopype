#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd
import string

from math import inf

from phenopype.settings import colours, flag_verbose, _image_viewer_arg_list, _annotation_function_dicts
from phenopype.utils_lowlevel import _create_mask_bool, _ImageViewer, _auto_line_width, _DummyClass

import phenopype.core.preprocessing as preprocessing


#%% functions



# pp.show_image(image)

def detect_contours(
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
    for rm in ["image", "kwargs"]:
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
        coord_list, support = [], []
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
                    coord_list.append(contour)
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
            print("- found " + str(len(coord_list)) + " contours that match criteria")
    else:
        if flag_verbose:
            print("- no contours found")

    ## return
    ret = {
    "info":{
        "annotation_type": "contour", 
        "pp_function": "detect_contours",
        },
    "settings": settings,
    "data":{
        "n_contours": len(coord_list),
        "coord_list": coord_list,
        "support": support,
        }
    }

    return ret


def edit_contours(
    image,
    annotation,
    **kwargs
):
    """
    Set points, draw a connected line between them, and measure its length. 

    Parameters
    ----------
    image : array
        input image
    annotation: dict
        annotation-dictionary containing contours 

    Returns
    -------
    (image_bin, annotation) : tuple
        modified image and contour modifications saved to annotations dictionary

    """
    ## kwargs
    contour_id = kwargs.get("contour_id")
    annotation_previous = kwargs.get("annotation_previous")
    
    ## retrieve settings for image viewer and for export
    ImageViewer_settings, local_settings  = {}, {}
    
    ## extract image viewer settings and data from previous annotations
    if "annotation_previous" in kwargs:       
        ImageViewer_previous = {}        
        ImageViewer_previous.update(annotation_previous["settings"])
        ImageViewer_previous["point_list"] = annotation_previous["data"]["point_list"]
        ImageViewer_previous = _DummyClass(ImageViewer_previous)   
        ImageViewer_settings["ImageViewer_previous"] = ImageViewer_previous

    ## check annotation dict input and convert to type/id/ann structure
    if list(annotation.keys())[0] == "info":
        if annotation["info"]["annotation_type"] == "contour":
            contours = annotation["data"]["coord_list"]
    else:
        if contour_id:
            local_settings["contour_id"] = contour_id
        else: 
            print("- contour_id missing - please provide contour ID [a-z]")
            return
        if list(annotation.keys())[0] in _annotation_function_dicts.keys():
            contours = annotation["contour"][contour_id]["data"]["coord_list"]
        elif list(annotation.keys())[0] in string.ascii_lowercase:
            contours = annotation[contour_id]["data"]["coord_list"]
            
    ## assemble image viewer settings from kwargs
    for key, value in kwargs.items():
        if key in _image_viewer_arg_list:
            ImageViewer_settings[key] = value
            local_settings[key] = value

    ## use GUI 
    out = _ImageViewer(image=image, 
                       contours=contours,
                       tool="draw", 
                       **ImageViewer_settings)
    
    ## checks
    if not out.done:
        print("- didn't finish: redo drawing")
        return 
    else:
        point_list = out.point_list
                
    ## extract modified binary mask
    image_bin = out.image_bin_copy
        
    ## return
    annotation = {
        "info": {
            "type": "drawing", 
            "function": "contour_modify",
            },
        "settings": local_settings,
        "data":{
            "point_list": point_list,
            }
        }
    return image_bin, annotation



def morphology(image, kernel_size=5, shape="rect", operation="close", iterations=1,
    **kwargs):
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

    ## method
    image = cv2.morphologyEx(
        image, op=operation_list[operation], kernel=kernel, iterations=iterations
    )

    ## return
    return image



def threshold(
    image,
    method="otsu",
    constant=1,
    blocksize=99,
    value=127,
    channel="gray",
    mask=None,
    **kwargs,
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
        if flag_verbose:
            image = preprocessing.select_channel(image, "gray")
    if blocksize % 2 == 0:
        if flag_verbose:
            blocksize = blocksize + 1
            print("- even blocksize supplied, adding 1 to make odd")

    ## method
    if method in "otsu":
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
    if not mask.__class__.__name__ == "NoneType":
        if not list(mask.keys())[0] == "a":
            mask = {"a": mask}
        mask_bool, include_idx, exclude_idx = np.zeros(thresh.shape, dtype=bool), 0,0
        for key, value in mask.items():
            coord_list, include = value["data"]["coord_list"], value["data"]["include"]
            if include == True:
                for coords in coord_list:
                    mask_bool = np.logical_or(mask_bool, _create_mask_bool(thresh, coords))
                    include_idx += 1
                thresh[mask_bool == False] = 0
            elif include == False:
                for coords in coord_list:
                    thresh[_create_mask_bool(thresh, coords)] = 0
                    exclude_idx += 1
        if exclude_idx>0:
            print("- excluding pixels from " + str(exclude_idx) + " drawn masks ")
        if include_idx>0:
            print("- including pixels from " + str(include_idx) + " drawn masks ")

    ## return
    return thresh



def watershed(
    image,
    image_thresh,
    iterations=1,
    kernel_size=3,
    distance_cutoff=0.8,
    distance_mask=0,
    distance_type="l1",
    **kwargs
):
    """
    Performs non-parametric marker-based segmentation - useful if many detected 
    contours are touching or overlapping with each other. Input image should be 
    a binary image that serves as the true background. Iteratively, edges are 
    eroded, the difference serves as markers.

    Parameters
    ----------
    image : array
        input image
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


    ## sure background
    sure_bg = morphology(
        image_thresh,
        operation="dilate",
        shape="ellipse",
        kernel_size=kernel_size,
        iterations=iterations,
    )
    
    ## distances in foreground 
    if distance_type in ["user", "l12", "fair", "welsch", "huber"]:
        distance_mask = 0
    opened = morphology(
        image_thresh,
        operation="erode",
        shape="ellipse",
        kernel_size=kernel_size,
        iterations=iterations,
    )
    
    ## distance transformation
    dist_transform = cv2.distanceTransform(
        opened, distance_type_list[distance_type], distance_mask
    )
    dist_transform = cv2.normalize(
        dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX
    )
    dist_transform = preprocessing.blur(dist_transform, kernel_size=int(2*kernel_size))
    dist_transform = abs(255 * (1-dist_transform)) 
    dist_transform = dist_transform.astype(np.uint8)

    ## sure foreground 
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
    markers = cv2.watershed(preprocessing.blur(image, int(2*kernel_size+1)), markers)

    ## convert to contours
    watershed_mask = np.zeros(image.shape[:2], np.uint8)
    watershed_mask[markers == -1] = 255
    watershed_mask[0 : watershed_mask.shape[0], 0] = 0
    watershed_mask[0 : watershed_mask.shape[0], watershed_mask.shape[1] - 1] = 0
    watershed_mask[0, 0 : watershed_mask.shape[1]] = 0
    watershed_mask[watershed_mask.shape[0] - 1, 0 : watershed_mask.shape[1]] = 0

    contours = detect_contours(watershed_mask, retrieval="ccomp")
    image_watershed = np.zeros(watershed_mask.shape, np.uint8)
           
    for coord, supp in zip(contours["data"]["coords"], contours["data"]["support"]):
        if supp["hierarchy_level"] == "child":
            cv2.drawContours(
                image=image_watershed,
                contours=[coord],
                contourIdx=0,
                thickness=-1,
                color=colours["white"],
                maxLevel=3,
                offset=None
                )
            cv2.drawContours(
                image=image_watershed,
                contours=[coord],
                contourIdx=0,
                thickness=2,
                color=colours["black"],
                maxLevel=3,
                offset=None
                )
    
    ## return
    return image_watershed

