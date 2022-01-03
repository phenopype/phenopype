#%% modules

import copy
import cv2
import numpy as np
import sys

from math import inf

from phenopype import __version__
from phenopype import settings 
from phenopype import utils_lowlevel
from phenopype.core import preprocessing


#%% functions

def detect_contour(
        image,
        approximation="simple",
        retrieval="ext",
        apply_drawing=True,
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
    A flavor of different approximation algorithms and retrieval intstructions can be applied. The contours can also be filtered for minimum area, diameter or nodes
    (= number of corners). 

    Parameters
    ----------
    image : ndarray
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
    annotation: dict
        phenopype annotation containing contours
    """    
    
    
    # =============================================================================
    # annotation management
    
    annotations = kwargs.get("annotations", {})
    
    ## drawings
    annotation_type = settings._drawing_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations, 
        annotation_type=annotation_type, 
        annotation_id=annotation_id, 
        kwargs=kwargs,
        prep_msg="- combining contours and drawing:"
    )

    ## contours
    fun_name = sys._getframe().f_code.co_name

    annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    # =============================================================================
    # setup
    
    image_bin = copy.deepcopy(image)

    if len(image_bin.shape) > 2:
        print("Multi-channel array supplied - need binary array.")
        return 
    
    if apply_drawing and "data" in annotation:
        
        drawings = annotation["data"][settings._drawing_type]
        
        ## apply coords to tool and draw on canvas
        for idx, coords in enumerate(drawings):
            if len(coords)==0:
                continue
            else:
                cv2.polylines(
                    image_bin,
                    np.array([coords[0]]),
                    False,
                    coords[1],
                    coords[2],
                )
              
	# =============================================================================
	# execute
        
    _ , contours_det, hierarchies_det = cv2.findContours(
        image=image_bin,
        mode=settings.opencv_contour_flags["retrieval"][retrieval],
        method=settings.opencv_contour_flags["approximation"][approximation],
        offset=tuple(offset_coords),
    )
   

	# =============================================================================
	# process
            
    if len(contours_det) > 0:
        contours, support = [], []
        for idx, (contour, hierarchy) in enumerate(zip(contours_det, hierarchies_det[0])):
            
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
                if all([
                        diameter > min_diameter and diameter < max_diameter,
                        area > min_area and area < max_area,
                            ]):
                
                    ## populate data lists
                    contours.append(contour)
                    support.append({
                            "center": center,
                            "area": area,
                            "diameter": diameter,
                            "hierarchy_level": hierarchy_level,
                            "hierarchy_idx_child": int(hierarchy[2]),
                            "hierarchy_idx_parent": int(hierarchy[3]),
                            })
                    
        if len(contours) == 0:
            print("- did not find any contours that match criteria")
            return
        else:
            print("- found " + str(len(contours)) + " contours that match criteria")
    else:
        print("- did not find any contours")
        return


	# =============================================================================
	# assemble results

    annotation = {
        "info":{
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            "annotation_type": annotation_type,
            },
        "settings": {
            "approximation":approximation,
            "retrieval":retrieval,
            "offset_coords":offset_coords,       
            "min_nodes":min_nodes,
            "max_nodes":max_nodes,
            "min_area":min_area,
            "max_area":max_area,
            "min_diameter":min_diameter,
            "max_diameter":max_diameter,
            },
        "data":{
            "n": len(contours),
            annotation_type: contours,
            "support": support,
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
    

def edit_contour(
    image,
    annotations,
    overlay_blend=0.2,
    overlay_line_width=1,
    overlay_colour_left="green",
    overlay_colour_right="red",
    **kwargs
):
    """
    Edit contours with a "paintbrush". The brush size can be controlled by pressing 
    Tab and using the mousewhell. Right-click removes, and left-click adds areas to 
    contours. Overlay colour, transparency (blend) and outline can be controlled.

    Parameters
    ----------
    image : ndarray
        input image
    overlay_blend: float, optional
        transparency / colour-mixing of the contour overlay 
    overlay_line_width: int, optional
        add outline to the contours. useful when overlay_blend == 0
    left_colour: str, optional
        overlay colour for left click (include). (for options see pp.colour)
    right_colour: str, optional
        overlay colour for right click (exclude). (for options see pp.colour)
        
    Returns
    -------
    annotations: dict
        phenopype annotation containing contours

    """
    # =============================================================================
    # annotation management
    
    ## get contours
    annotation_type = settings._contour_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations, 
        annotation_type=annotation_type, 
        annotation_id=annotation_id, 
        kwargs=kwargs,
    )
    
    gui_data = {annotation_type: utils_lowlevel._get_GUI_data(annotation)}
        
    
    ## get previous drawing
    fun_name = sys._getframe().f_code.co_name

    annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)
    
    annotation = utils_lowlevel._get_annotation(
        annotations=annotations, 
        annotation_type=annotation_type, 
        annotation_id=annotation_id, 
        kwargs=kwargs,
    )
    
    gui_data.update({settings._sequence_type: utils_lowlevel._get_GUI_data(annotation)})
    gui_settings = utils_lowlevel._get_GUI_settings(kwargs, annotation)
    

    # =============================================================================
    # setup    

    overlay_colour_left = utils_lowlevel._get_bgr(overlay_colour_left)     
    overlay_colour_right = utils_lowlevel._get_bgr(overlay_colour_right)   
    
	# =============================================================================
	# execute
        
    gui = utils_lowlevel._GUI(
        image=image, 
        tool="draw", 
        overlay_blend=overlay_blend,
        overlay_line_width=overlay_line_width,
        overlay_colour_left=overlay_colour_left,
        overlay_colour_right=overlay_colour_right,
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
            "overlay_blend": overlay_blend,
            "overlay_line_width": overlay_line_width,
            "overlay_colour_left": overlay_colour_left,
            "overlay_colour_right": overlay_colour_right,
            },
        "data":{
            annotation_type: gui.data[settings._sequence_type],
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


def morphology(
        image, 
        kernel_size=5, 
        shape="rect", 
        operation="close", 
        iterations=1,
        **kwargs
        ):
    """
    Performs advanced morphological transformations using erosion and dilation 
    as basic operations. Provides different kernel shapes and a suite of operation
    types (read more about morphological operations here:
    https://docs.opencv.org/master/db/df6/tutorial_erosion_dilatation.html)

    Parameters
    ----------
    image : ndarray
        input image (binary)
    kernel_size: int, optional
        size of the morphology kernel (has to be odd - even numbers will be ceiled)
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
    image : ndarray
        processed binary image

    """
	# =============================================================================
	# setup 
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
        
        
	# =============================================================================
	# execute
        
    kernel = cv2.getStructuringElement(settings.opencv_morphology_flags["shape_list"][shape], 
                                       (kernel_size, kernel_size))
    operation = settings.opencv_morphology_flags["operation_list"][operation]
    
    image = cv2.morphologyEx(
        image, op=operation, kernel=kernel, iterations=iterations
    )
    

	# =============================================================================
	# return

    ## return
    return image



def threshold(
    image,
    method="otsu",
    constant=1,
    blocksize=99,
    value=127,
    channel=None,
    invert=False,
    **kwargs,
):
    """
    Applies a threshold to create a binary image from a grayscale or a multichannel 
    image (see phenopype.core.preprocessing.decompose_image for channel options).
    
    Three types of thresholding algorithms are supported: 
        - otsu: use Otsu algorithm to choose the optimal threshold value
        - adaptive: dynamic threshold values across image (uses arguments
          "blocksize" and "constant")
        - binary: fixed threshold value (uses argument "value")    
        
    Mask annotations can be supplied to include or exclude areas.

    Parameters
    ----------
    image : ndarray
        input image

    method : {"otsu", "adaptive", "binary"} str, optional
        type of thresholding algorithm to be used
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

    Returns
    -------
    image : ndarray
        binary image

    """


	# =============================================================================
	# setup 
    
    verbose = kwargs.get("verbose", settings.flag_verbose)
    
    if len(image.shape) == 3:
        if not channel:
            channel = "gray"
            print("- multichannel image supplied, converting to grayscale")
        image = preprocessing.decompose_image(image, channel, verbose=verbose)
            
    if blocksize % 2 == 0:
        blocksize = blocksize + 1
        print("- even blocksize supplied, adding 1 to make odd")
        
    if invert:
        image = cv2.bitwise_not(image)

	# =============================================================================
	# execute
    
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
        
    # =============================================================================
    # annotation management
    
    annotations = kwargs.get("annotations", {})
        
    ## references
    annotation_id_ref = kwargs.get(settings._reference_type + "_id", None)
    annotation_ref = utils_lowlevel._get_annotation(
        annotations, settings._reference_type, annotation_id_ref,
        prep_msg = "- masking regions in thresholded image:")

    ## masks 
    annotation_id_mask= kwargs.get(settings._mask_type + "_id", None)
    annotation_mask = utils_lowlevel._get_annotation(
        annotations, settings._mask_type, annotation_id_mask,
        prep_msg = "- masking regions in thresholded image:")
        
    # =============================================================================
    # execute masking

            
    if "data" in annotation_mask:
        if settings._mask_type in annotation_mask["data"]:
            mask_bool, include_idx, exclude_idx = np.zeros(thresh.shape, dtype=bool), 0,0
        
            polygons = annotation_mask["data"][settings._mask_type]
            include = annotation_mask["data"]["include"]
    
            if include == True:
                for coords in polygons:
                    mask_bool = np.logical_or(mask_bool, utils_lowlevel._create_mask_bool(thresh, coords))
                    include_idx += 1
                thresh[mask_bool == False] = 0
            elif include == False:
                for coords in polygons:
                    thresh[utils_lowlevel._create_mask_bool(thresh, coords)] = 0
                    exclude_idx += 1
                    
            if exclude_idx>0 and verbose:
                print("- excluding pixels from " + str(exclude_idx) + " drawn masks ")
            if include_idx>0 and verbose:
                print("- including pixels from " + str(include_idx) + " drawn masks ")
                        
    if "data" in annotation_ref:
        if settings._mask_type in annotation_ref["data"]:
            polygons = annotation_ref["data"][settings._mask_type]
            for coords in polygons:
                thresh[utils_lowlevel._create_mask_bool(thresh, coords)] = 0
            print("- excluding pixels from reference")


	# =============================================================================
	# return

    return thresh



def watershed(
    image,
    annotations,
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
    image : ndarray
        input image
    image_thresh: array
        binary image (e.g. from threshold)
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
    image : ndarray
        binary image

    """
    # =============================================================================
    # annotation management
    
    ## get contours
    annotation_type = settings._contour_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations, 
        annotation_type=annotation_type, 
        annotation_id=annotation_id, 
        kwargs=kwargs,
    )
    
    contours = annotation["data"][settings._contour_type]

	# =============================================================================
	# setup 

    if len(contours) > 0:

        ## coerce to multi channel image for colour mask
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        ## create binary overlay
        image_bin = np.zeros(image.shape[0:2], dtype=np.uint8)
        
        ## draw contours onto overlay
        for contour in contours:
            cv2.drawContours(
                image=image_bin,
                contours=[contour],
                contourIdx=0,
                thickness=-1,
                color=255,
                maxLevel=3,
                offset=(0,0),
                )
    
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

	# =============================================================================
	# execute

    ## sure background
    sure_bg = morphology(
        image_bin,
        operation="dilate",
        shape="ellipse",
        kernel_size=kernel_size,
        iterations=iterations,
    )
    
    ## distances in foreground 
    if distance_type in ["user", "l12", "fair", "welsch", "huber"]:
        distance_mask = 0
        
    opened = morphology(
        image_bin,
        operation="erode",
        shape="ellipse",
        kernel_size=kernel_size,
        iterations=iterations,
    )
    
    ## distance transformation
    dist_transform = cv2.distanceTransform(
        opened, settings.opencv_distance_flags[distance_type], distance_mask
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

    annotations_contour = detect_contour(watershed_mask, retrieval="ccomp")
    image_watershed = np.zeros(watershed_mask.shape, np.uint8)

    for coord, supp in zip(annotations_contour[annotation_type]["a"]["data"][annotation_type], 
                           annotations_contour[annotation_type]["a"]["data"]["support"]):
        if supp["hierarchy_level"] == "child":
            cv2.drawContours(
                image=image_watershed,
                contours=[coord],
                contourIdx=0,
                thickness=-1,
                color=utils_lowlevel._get_bgr("white"),
                maxLevel=3,
                offset=None
                )
            cv2.drawContours(
                image=image_watershed,
                contours=[coord],
                contourIdx=0,
                thickness=2,
                color=utils_lowlevel._get_bgr("black"),
                maxLevel=3,
                offset=None
                )
    
	# =============================================================================
	# return
    
    return image_watershed



# def watershed(
#     image,
#     annotations,
#     iterations=1,
#     kernel_size=3,
#     distance_cutoff=0.8,
#     distance_mask=0,
#     distance_type="l1",
#     **kwargs
# ):
#     """
#     Performs non-parametric marker-based segmentation - useful if many detected 
#     contours are touching or overlapping with each other. Input image should be 
#     a binary image that serves as the true background. Iteratively, edges are 
#     eroded, the difference serves as markers.

#     Parameters
#     ----------
#     image : ndarray
#         input image
#     image_thresh: array
#         binary image (e.g. from threshold)
#     kernel_size: int, optional
#         size of the diff-kernel (has to be odd - even numbers will be ceiled)
#     iterations : int, optional
#         number of times to apply diff-operation
#     distance_cutoff : {between 0 and 1} float, optional
#         watershed distance transform cutoff (larger values discard more pixels)
#     distance_mask : {0, 3, 5} int, optional
#         size of distance mask - not all sizes work with all distance types (will
#         be coerced to 0)
#     distance_type : {"user", "l1", "l2", "C", "l12", "fair", "welsch", "huber"} str, optional
#         distance transformation type

#     Returns
#     -------
#     image : ndarray
#         binary image

#     """
#     # =============================================================================
#     # annotation management
    
#     ## get contours
#     annotation_type = settings._contour_type
#     annotation_id = kwargs.get(annotation_type + "_id", None)

#     annotation = utils_lowlevel._get_annotation(
#         annotations=annotations, 
#         annotation_type=annotation_type, 
#         annotation_id=annotation_id, 
#         kwargs=kwargs,
#     )
    
#     contours = annotation["data"][settings._contour_type]

# 	# =============================================================================
# 	# setup 

#     if len(contours) > 0:

#         ## coerce to multi channel image for colour mask
#         if len(image.shape) == 2:
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
#         ## create binary overlay
#         image_bin = np.zeros(image.shape[0:2], dtype=np.uint8)
        
#         ## draw contours onto overlay
#         for contour in contours:
#             cv2.drawContours(
#                 image=image_bin,
#                 contours=[contour],
#                 contourIdx=0,
#                 thickness=-1,
#                 color=255,
#                 maxLevel=3,
#                 offset=(0,0),
#                 )
    

    
#     if kernel_size % 2 == 0:
#         kernel_size = kernel_size + 1

# 	# =============================================================================
# 	# execute

#     ## sure background
#     sure_bg = morphology(
#         image_bin,
#         operation="dilate",
#         shape="ellipse",
#         kernel_size=kernel_size,
#         iterations=iterations,
#     )
    
#     ## distances in foreground 
#     if distance_type in ["user", "l12", "fair", "welsch", "huber"]:
#         distance_mask = 0
        
#     opened = morphology(
#         image_bin,
#         operation="erode",
#         shape="ellipse",
#         kernel_size=kernel_size,
#         iterations=iterations,
#     )
    
#     ## distance transformation
#     dist_transform = cv2.distanceTransform(
#         opened, settings.opencv_distance_flags[distance_type], distance_mask
#     )
#     dist_transform = cv2.normalize(
#         dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX
#     )
#     dist_transform = preprocessing.blur(dist_transform, kernel_size=int(2*kernel_size))
#     dist_transform = abs(255 * (1-dist_transform)) 
#     dist_transform = dist_transform.astype(np.uint8)

#     ## sure foreground 
#     sure_fg = threshold(dist_transform, method="binary", value=int(distance_cutoff*255))

#     ## finding unknown region
#     sure_fg = sure_fg.astype("uint8")
#     sure_fg[sure_fg == 1] = 255
#     unknown = cv2.subtract(sure_bg, sure_fg)

#     ## marker labelling
#     ret, markers = cv2.connectedComponents(sure_fg)
#     markers = markers + 1
#     markers[unknown == 255] = 0

#     ## watershed
#     markers = cv2.watershed(preprocessing.blur(image, int(2*kernel_size+1)), markers)

#     ## convert to contours
#     watershed_mask = np.zeros(image.shape[:2], np.uint8)
#     watershed_mask[markers == -1] = 255
#     watershed_mask[0 : watershed_mask.shape[0], 0] = 0
#     watershed_mask[0 : watershed_mask.shape[0], watershed_mask.shape[1] - 1] = 0
#     watershed_mask[0, 0 : watershed_mask.shape[1]] = 0
#     watershed_mask[watershed_mask.shape[0] - 1, 0 : watershed_mask.shape[1]] = 0

#     annotations_contour = detect_contour(watershed_mask, retrieval="ccomp")
#     image_watershed = np.zeros(watershed_mask.shape, np.uint8)

#     for coord, supp in zip(annotations_contour[annotation_type]["a"]["data"][annotation_type], 
#                            annotations_contour[annotation_type]["a"]["data"]["support"]):
#         if supp["hierarchy_level"] == "child":
#             cv2.drawContours(
#                 image=image_watershed,
#                 contours=[coord],
#                 contourIdx=0,
#                 thickness=-1,
#                 color=utils_lowlevel._get_bgr("white"),
#                 maxLevel=3,
#                 offset=None
#                 )
#             cv2.drawContours(
#                 image=image_watershed,
#                 contours=[coord],
#                 contourIdx=0,
#                 thickness=2,
#                 color=utils_lowlevel._get_bgr("black"),
#                 maxLevel=3,
#                 offset=None
#                 )
    
# 	# =============================================================================
# 	# return
    
#     return image_watershed

