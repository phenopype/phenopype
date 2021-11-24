#%% modules
import cv2, copy
import math
import numpy as np
import numpy.ma as ma
import pandas as pd
from dataclasses import make_dataclass
import string 

import logging
from radiomics import featureextractor
import SimpleITK as sitk
from tqdm import tqdm as _tqdm



from phenopype.utils_lowlevel import (
    _ImageViewer,
    _auto_line_width,
    _auto_point_size,
    _auto_text_width,
    _auto_text_size,
    _drop_dict_entries,
    _load_previous_annotation,
    _provide_annotation_data,
    _update_settings,
)
from phenopype.settings import (
    colours, 
    flag_verbose, 
    _image_viewer_arg_list, 
    _annotation_types,
    opencv_skeletonize_flags,
    )

from phenopype.core.preprocessing import decompose_image


#%% methods


def set_landmark(
    image,
    point_colour="green",
    point_size="auto",
    label=True,
    label_colour="black",
    label_size="auto",
    label_width="auto",
    **kwargs
):
    """
    Placing landmarks. It is possible to modify the appearance of points while 
    while doing so. 
    
    Parameters
    ----------

    point_colour: {"green", "red", "blue", "black", "white"} str, optional
        landmark point colour
    point_size: int, optional
        landmark point size in pixels
    label_colour : {"black", "white", "green", "red", "blue"} str, optional
        landmark label colour.
    label_size: int, optional
        landmark label font size (scaled to image)
    label_width: int, optional
        landmark label font width  (scaled to image)

    Returns
    -------
    df_masks: DataFrame or container
        contains landmark coordiantes
    """

	# =============================================================================
	# setup 
    
    annotation_previous = kwargs.get("annotation_previous", None)


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
                ("data","points"),
                ])            
        
    ## update local and IV settings from kwargs
    if kwargs:
        _update_settings(kwargs, local_settings, IV_settings)
        
        
	# =============================================================================
	# further prep

    ## configure points
    if point_size == "auto":
        point_size = _auto_point_size(image)
    if label_size == "auto":
        label_size = _auto_text_size(image)
    if label_width == "auto":
        label_width = _auto_text_width(image)


	# =============================================================================
	# execute

    out = _ImageViewer(image=image, 
                       tool="point", 
                       flag_text_label=label,
                       point_size=point_size,
                       point_colour=point_colour,
                       label_size=label_size,
                       label_width=label_width,
                       label_colour=label_colour,                       
                       **IV_settings)
    
    ## checks
    if not out.done:
        print("- didn't finish: redo landmarks")
        return 
    elif len(out.points) == 0:
        print("- zero coordinates: redo landmarks")
        return 
    else:
        points = out.points
        
        
	# =============================================================================
	# assemble results

    annotation = {
        "info": {
            "annotation_type": "landmark", 
            "function": "set_landmark",
            },
        "settings": local_settings,
        "data":{
            "points": points,
            }
        }
    
    
	# =============================================================================
	# return
    
    return annotation


def set_polyline(
    image,
    line_width="auto",
    line_colour="green",
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
    overwrite: bool, optional
        if working using a container, or from a phenopype project directory, 
        should existing polylines be overwritten
    line_width: int, optional
        width of polyline

    Returns
    -------
    df_polylines : DataFrame or container
        contains the drawn polylines

    """
	# =============================================================================
	# setup 
    
    ## kwargs
    annotation_previous = kwargs.get("annotation_previous", None)
    

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
        
    print(kwargs)

	# =============================================================================
	# further prep
    
    if line_width == "auto":
        line_width = _auto_line_width(image)


    ## method
    out = _ImageViewer(image=image,
                       tool="polyline",
                       line_width=line_width,
                       line_colour=line_colour,
                       **IV_settings)
    
    ## checks
    if not out.done:
        print("- didn't finish: redo landmarks")
        return 
    elif len(out.coord_list) == 0:
        print("- zero coordinates: redo landmarks")
        return 
    else:
        coord_list = out.coord_list
        

	# =============================================================================
	# assemble results
    
    annotation = {
        "info": {
            "annotation_type": "line", 
            "function": "set_polyline",
            },
        "settings": local_settings,
        "data":{
            "coord_list": coord_list,
            }
        }
    
	# =============================================================================
	# return
    
    return annotation



def skeletonize(
    image, 
    annotation, 
    thinning="zhangsuen", 
    **kwargs,
):
    """
    Applies a binary blob thinning operation, to achieve a skeletization of 
    the input image using the technique, i.e. retrieve the topological skeleton
    (https://en.wikipedia.org/wiki/Topological_skeleton), using the algorithms 
    of Thang-Suen or Guo-Hall.

    Parameters
    ----------
    obj_input : array or container
        input object (binary)
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata, will be added to contour
        output DataFrame
    df_contours : DataFrame, optional
        contains contours
    thinning: {"zhangsuen", "guohall"} str, optional
        type of thinning algorithm to apply

    Returns
    -------
    image : array or container
        thinned binary image
    """

    # =============================================================================
    # retain settings
    
    padding = kwargs.get("padding", 1)

    ## retrieve settings from args
    local_settings  = _drop_dict_entries(
        locals(), drop=["image", "kwargs", "annotation"])
        
    ## update settings from kwargs
    if kwargs:
        _update_settings(kwargs, local_settings)
        

	# =============================================================================
	# setup 

    ## extract annotation data     
    contours = _provide_annotation_data(annotation, 
                                        "contour",
                                        "coord_list",
                                        kwargs)

    if not contours:
        return {}
        
	# =============================================================================
	# execute
    
    coord_list = []
    
    for coords in contours:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        image_sub = image[
            (ry - padding) : (ry + rh + padding), (rx - padding) : (rx + rw + padding)
        ]

        mask = np.zeros(image_sub.shape[0:2], np.uint8)
        mask = cv2.fillPoly(mask, [coords], 255, offset=(-rx + padding, -ry + padding))

        skeleton = cv2.ximgproc.thinning(mask, thinningType=opencv_skeletonize_flags[thinning])
        skel_ret, skel_contour, skel_hierarchy = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        skel_contour = skel_contour[0]
        skel_contour[:, :, 0] = skel_contour[:, :, 0] + rx - padding
        skel_contour[:, :, 1] = skel_contour[:, :, 1] + ry - padding
                
        coord_list.append(skel_contour)
               
        
	# =============================================================================
	# assemble results
    
    annotation = {
        "info": {
            "annotation_type": "line", 
            "function": "skeletonize",
            },
        "settings": local_settings,
        "data":{
            "coord_list": coord_list,
            }
        }
    
    
	# =============================================================================
	# return
    
    return annotation
    


def shape_features(
    annotation,
    basic=True,
    moments=False, 
    hu_moments=False,
    **kwargs
):
    """
    Collects a set of 41 shape descriptors from every contour. There are three sets of 
    descriptors: basic shape descriptors, moments, and hu moments. Two additional features,
    contour area and diameter are already provided by the find_contours function.
    https://docs.opencv.org/3.4.9/d3/dc0/group__imgproc__shape.html

    Of the basic shape descriptors, all 12 are translational invariants, 8 are rotation 
    invariant (rect_height and rect_width are not) and  4 are also scale invariant 
    (circularity, compactness, roundness, solidity).
    https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)  
                                
    The moments set encompasses 10 raw spatial moments (some are translation and rotation
    invariants, but not all), 7 central moments (all translational invariant) and 7 central 
    normalized moments (all translational and scale invariant).
    https://en.wikipedia.org/wiki/Image_moment
    
    The 7 hu moments are derived of the central moments, and are all translation, scale 
    and rotation invariant.
    http://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Hu.pdf
        
    Basic shape descriptors:
        circularity = 4 * np.pi * contour_area / contour_perimeter_length^2
        compactness = √(4 * contour_area / pi) / contour_diameter
        min_rect_max = minimum bounding rectangle major axis
        min_rect_min = minimum bounding rectangle minor axis
        perimeter_length = total length of contour perimenter
        rect_height = height of the bounding rectangle ("caliper dim 1")
        rect_width = width of the bounding rectangle ("caliper dim 2")
        roundness = (4 * contour_area) / pi * contour_perimeter_length^2
        solidity = contour_area / convex_hull_area
        tri_area = area of minimum bounding triangle

    Moments:
        raw moments = m00, m10, m01, m20, m11, m02, m30, m21,  m12, m03
        central moments = mu20, mu11, mu02, mu30, mu21, mu12, mu03,  
        normalized central moments = nu20, nu11, nu02, nu30, nu21, nu12, nu03

    Hu moments:
        hu moments = hu1, hu2, hu3, hu4, hu5, hu6, hu7

    Parameters
    ----------
    obj_input : array or container
        input object
    df_contours : DataFrame, optional
        contains the contours
    return_basic: True, opational
        append the basic shape descriptors to a provided contour DataFrame
    return_moments: False, optional
        append the basic shape descriptors to a provided contour DataFrame
    return_hu_moments: False, optional
        append the basic shape descriptors to a provided contour DataFrame
        
    Returns
    -------
    df_contours : DataFrame or container
        contains contours, and added features

    """
    
    # =============================================================================
    # retain settings
    
    contour_id = kwargs.get("contour_id")
    
    # kwargs
    flags = make_dataclass(cls_name="flags", 
                           fields=[("basic", bool, basic),
                                   ("moments", bool, moments), 
                                   ("hu_moments", bool, hu_moments)])

    ## retrieve settings from args
    local_settings  = _drop_dict_entries(
        locals(), drop=["kwargs", "annotation", "flags"])
        
    ## update settings from kwargs
    if kwargs:
        _update_settings(kwargs, local_settings)
        

    # =============================================================================
    # setup             

    contours = _provide_annotation_data(annotation, "contour", "coord_list", kwargs)
    contours_support = _provide_annotation_data(annotation, "contour", "support", kwargs)
    
    if not contours or not contours_support:
        return {}

    # =============================================================================
    # execute
    
    shape_features = {}
    
    for idx1, (coords, support) in enumerate(zip(contours, contours_support)):      
        
        idx1 += 1
        shape_features[idx1] = {}
        
        ## basic shape descriptors
        if flags.basic:

            ## retrieve area and diameter, needed for calculation
            cnt_diameter = support["diameter"]
            cnt_area = support["area"]

            tri_area, tri_coords = cv2.minEnclosingTriangle(coords)
            min_rect_center, min_rect_min_max, min_rect_angle = cv2.minAreaRect(coords)
            min_rect_max, min_rect_min = min_rect_min_max[0], min_rect_min_max[1]
            rect_x, rect_y, rect_width, rect_height = cv2.boundingRect(coords)
            perimeter_length = cv2.arcLength(coords, closed=True)
            circularity = 4 * np.pi * cnt_area / math.pow(perimeter_length, 2)
            roundness = (4 * cnt_area) / (np.pi * math.pow(cnt_diameter, 2))
            solidity = cnt_area / cv2.contourArea(cv2.convexHull(coords))
            compactness = math.sqrt(4 * cnt_area / np.pi) / cnt_diameter
            
            basic = {
                'area': cnt_area,
                'circularity': circularity, 
                'diatmeter': cnt_diameter,
                'compactness': compactness,
                'min_rect_max': min_rect_max,
                'min_rect_min': min_rect_min,
                'perimeter_length': perimeter_length,
                'rect_height': rect_height,
                'rect_width': rect_width,
                'roundness': roundness,
                'solidity':solidity,
                'tri_area': tri_area,
                }
            
            shape_features[idx1] = {**shape_features[idx1], **basic}
                                             
        ## moments
        if flags.moments:
            moments = cv2.moments(coords)
            shape_features[idx1] = {**shape_features[idx1], **moments}


        ## hu moments
        if flags.hu_moments:
            hu_moments = {}
            for idx2, mom in enumerate(cv2.HuMoments(moments)):
                hu_moments["hu" + str(idx2 + 1)] = mom[0]
            shape_features[idx1] = {**shape_features[idx1], **hu_moments}

	# =============================================================================
	# return
        
    annotation = {
        "info": {
            "annotation_type": "morphology", 
            "function": "shape_features",
            },
        "settings": local_settings,
        "data":{
            "features": shape_features,
            }
        }
    
    return annotation


def texture_features(
    image,
    annotation,
    channels=["gray"],
    background=False,
    background_ext=20,
    min_diameter=5,
    features = ["firstorder"],
    **kwargs
):
    """
    Collects 120 texture features using the pyradiomics feature extractor
    (https://pyradiomics.readthedocs.io/en/latest/features.html): 
    
    - First Order Statistics (19 features)
    - Shape-based (3D) (16 features)
    - Shape-based (2D) (10 features)
    - Gray Level Cooccurence Matrix (24 features)
    - Gray Level Run Length Matrix (16 features)
    - Gray Level Size Zone Matrix (16 features)
    - Neighbouring Gray Tone Difference Matrix (5 features)
    - Gray Level Dependence Matrix (14 features)
    
    Features are collected from EVERY contour that is supplied along with the raw
    image. Not that this may result in very large dataframes. 

    Parameters
    ----------
    obj_input : array or container
        input object
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata, will be added to
        output DataFrame
    df_contours : DataFrame, optional
        contains the contours
    channels : {"gray", "rgb"} str, optional
        for which channels should pixel intensity be measured
    background: bool, optional
        measure the pixels of the background in an extended (background_ext) 
        bounding box around the contour
    background_ext: in, optional
        value in pixels by which the bounding box should be extended
    min_diameter: int, optional
        minimum diameter of the contour
    features: list, optional
        firstorder, shape, glcm, gldm, glrlm, glszm, ngtdm]

    Returns
    -------
    df_textures : DataFrame or container
        contains the pixel intensities 

    """
    
    # =============================================================================
    # retain settings
    
    contour_id = kwargs.get("contour_id")
    
    ## retrieve settings from args
    local_settings  = _drop_dict_entries(
        locals(), drop=["image","kwargs", "annotation"])
        
    ## update settings from kwargs
    if kwargs:
        _update_settings(kwargs, local_settings)
        
    if not isinstance(channels, list):
        channels = [channels]

    feature_activation = {}
    for feature in features:
        feature_activation[feature] = []
        
        

	# =============================================================================
	# setup 
       
    contours = _provide_annotation_data(annotation, "contour", "coord_list", kwargs)
    contours_support = _provide_annotation_data(annotation, "contour", "support", kwargs)
    
    if not contours or not contours_support:
        return {}
    
    
	# =============================================================================
	# execute

    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    ## create forgeround mask
    foreground_mask_inverted = np.zeros(image.shape[:2], np.uint8)
    for coords in contours:
        foreground_mask_inverted = cv2.fillPoly(
            foreground_mask_inverted, [coords], 255
        )

    
    texture_features = {}
    
    for channel in channels:

        image_slice = decompose_image(image, channel)
        texture_features[channel] = {}

        for idx1, (coords, support) in _tqdm(
                enumerate(zip(contours, contours_support)),
                "Processing " + channel + " channel texture features",
                total=len(contours)):
            
            idx1 += 1
            
            if support["diameter"] > min_diameter:
                
                rx, ry, rw, rh = cv2.boundingRect(coords)

                data=image_slice[ry : ry + rh, rx : rx + rw]
                mask=foreground_mask_inverted[ry : ry + rh, rx : rx + rw]
                sitk_data = sitk.GetImageFromArray(data)
                sitk_mask = sitk.GetImageFromArray(mask)
                
                extractor = featureextractor.RadiomicsFeatureExtractor()
                extractor.disableAllFeatures()
                extractor.enableFeaturesByName(**feature_activation)
                features = extractor.execute(sitk_data, sitk_mask, label=255)
                
                output = {}
                for key, val in features.items():
                    if not "diagnostics" in key :
                        output[key.split('_', 1)[1]  ] = float(val)
                        
                texture_features[channel][idx1] = output 

	# =============================================================================
	# return
        
    annotation = {
        "info": {
            "annotation_type": "texture", 
            "function": "texture_features",
            },
        "settings": local_settings,
        "data":{
            "features": texture_features,
            }
        }
    
    return annotation
