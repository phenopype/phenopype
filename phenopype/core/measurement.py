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

from phenopype import utils_lowlevel
from phenopype import settings
from phenopype.core import preprocessing 


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
    Place landmarks. Note that modifying the appearance of the points will only 
    be effective for the placement, not for subsequent drawing, visualization, 
    and export.
    
    Parameters
    ----------
    image : ndarray
        input image
    point_colour: str, optional
        landmark point colour (for options see pp.colour)
    point_size: int, optional
        landmark point size in pixels
    label : bool, optional
        add text label
    label_colour : str, optional
        landmark label colour (for options see pp.colour)
    label_size: int, optional
        landmark label font size (scaled to image)
    label_width: int, optional
        landmark label font width  (scaled to image)

    Returns
    -------
    annotations: dict
        phenopype annotation containing landmarks
    """

	# =============================================================================
	# setup 
    
    annotation_previous = kwargs.get("annotation_previous", None)


    # =============================================================================
    # retain settings

    ## retrieve settings from args
    local_settings  = utils_lowlevel._drop_dict_entries(locals(),
        drop=["image","kwargs","annotation_previous"])

    ## retrieve update IV settings and data from previous annotations  
    IV_settings = {}     
    if annotation_previous:       
        IV_settings["ImageViewer_previous"] =utils_lowlevel._load_previous_annotation(
            annotation_previous = annotation_previous, 
            components = [
                ("data","points"),
                ])            
        
    ## update local and IV settings from kwargs
    if kwargs:
        utils_lowlevel._update_settings(kwargs, local_settings, IV_settings)
        
        
	# =============================================================================
	# further prep

    ## configure points
    if point_size == "auto":
        point_size = utils_lowlevel._auto_point_size(image)
    if label_size == "auto":
        label_size = utils_lowlevel._auto_text_size(image)
    if label_width == "auto":
        label_width = utils_lowlevel._auto_text_width(image)


	# =============================================================================
	# execute

    out = utils_lowlevel._ImageViewer(
        image=image, 
        tool="point", 
        flag_text_label=label,
        point_size=point_size,
        point_colour=point_colour,
        label_size=label_size,
        label_width=label_width,
        label_colour=label_colour,                       
        **IV_settings,
        )
    
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
    Set points, draw a connected line between them, and measure its length. Note 
    that modifying the appearance of the lines will only be effective for the 
    placement, not for subsequent drawing, visualization, and export.

    Parameters
    ----------
    image : ndarray
        input image
    line_width: int, optional
        width of polyline
    line_colour : str, optional
        poly line colour (for options see pp.colour)
        
    Returns
    -------
    annotations: dict
        phenopype annotation containing polylines

    """
	# =============================================================================
	# setup 
    
    ## kwargs
    annotation_previous = kwargs.get("annotation_previous", None)
    

    # =============================================================================
    # retain settings

    ## retrieve settings from args
    local_settings  = utils_lowlevel._drop_dict_entries(locals(),
        drop=["image","kwargs","annotation_previous"])

    ## retrieve update IV settings and data from previous annotations  
    IV_settings = {}     
    if annotation_previous:       
        IV_settings["ImageViewer_previous"] =utils_lowlevel._load_previous_annotation(
            annotation_previous = annotation_previous, 
            components = [
                ("data","coord_list"),
                ])            
        
    ## update local and IV settings from kwargs
    if kwargs:
        utils_lowlevel._update_settings(kwargs, local_settings, IV_settings)
        
    print(kwargs)

	# =============================================================================
	# further prep
    
    if line_width == "auto":
        line_width = utils_lowlevel._auto_line_width(image)


    ## method
    out = utils_lowlevel._ImageViewer(image=image,
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
    image : ndarray
        input image
    annotation: dict
        phenopype annotation containing contours
    thinning: {"zhangsuen", "guohall"} str, optional
        type of thinning algorithm to apply
        
    Returns
    -------
    annotations: dict
        phenopype annotation containing skeleton coords
    """

    # =============================================================================
    # retain settings
    
    padding = kwargs.get("padding", 1)

    ## retrieve settings from args
    local_settings  = utils_lowlevel._drop_dict_entries(
        locals(), drop=["image", "kwargs", "annotation"])
        
    ## update settings from kwargs
    if kwargs:
        utils_lowlevel._update_settings(kwargs, local_settings)
        

	# =============================================================================
	# setup 

    ## extract annotation data     
    contours = utils_lowlevel._provide_annotation_data(annotation, 
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

        skeleton = cv2.ximgproc.thinning(mask, thinningType=settings.opencv_skeletonize_flags[thinning])
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
    features=["basic"],
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
        - circularity = 4 * np.pi * contour_area / contour_perimeter_length^2
        - compactness = √(4 * contour_area / pi) / contour_diameter
        - min_rect_max = minimum bounding rectangle major axis
        - min_rect_min = minimum bounding rectangle minor axis
        - perimeter_length = total length of contour perimenter
        - rect_height = height of the bounding rectangle ("caliper dim 1")
        - rect_width = width of the bounding rectangle ("caliper dim 2")
        - roundness = (4 * contour_area) / pi * contour_perimeter_length^2
        - solidity = contour_area / convex_hull_area
        - tri_area = area of minimum bounding triangle

    Moments:
        - raw moments = m00, m10, m01, m20, m11, m02, m30, m21,  m12, m03
        - central moments = mu20, mu11, mu02, mu30, mu21, mu12, mu03,  
        - normalized central moments = nu20, nu11, nu02, nu30, nu21, nu12, nu03

    Hu moments:
        hu moments:
            hu1 - hu7

    Parameters
    ----------
    image : ndarray
        input image
    features: ["basic", "moments", "hu_moments"]    
        type of shape features to extract
        
    Returns
    -------
    annotations: dict
        phenopype annotation containing shape features

    """
    
    # =============================================================================
    # retain settings
    
    contour_id = kwargs.get("contour_id")

    ## retrieve settings from args
    local_settings  = utils_lowlevel._drop_dict_entries(
        locals(), drop=["kwargs", "annotation"])
        
    ## update settings from kwargs
    if kwargs:
        utils_lowlevel._update_settings(kwargs, local_settings)
        

    # =============================================================================
    # setup             

    contours = utils_lowlevel._provide_annotation_data(annotation, "contour", "coord_list", kwargs)
    contours_support = utils_lowlevel._provide_annotation_data(annotation, "contour", "support", kwargs)
    
    if not contours or not contours_support:
        return {}

    # =============================================================================
    # execute
    
    shape_features = {}
    
    for idx1, (coords, support) in enumerate(zip(contours, contours_support)):      
        
        idx1 += 1
        shape_features[idx1] = {}
        
        ## basic shape descriptors
        if "basic" in features:

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
                'diameter': cnt_diameter,
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
        if "moments" in features:
            moments = cv2.moments(coords)
            shape_features[idx1] = {**shape_features[idx1], **moments}


        ## hu moments
        if "hu_moments" in features:
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
    features = ["firstorder"],
    channels=["gray"],
    min_diameter = 5,
    **kwargs
):
    """
    Collects 120 texture features using the pyradiomics feature extractor
    ( https://pyradiomics.readthedocs.io/en/latest/features.html ): 
    
    - firstorder: First Order Statistics (19 features)
    - shape2d: Shape-based (2D) (16 features)
    - glcm: Gray Level Cooccurence Matrix (24 features)
    - gldm: Gray Level Dependence Matrix (14 features)
    - glrm: Gray Level Run Length Matrix (16 features)
    - glszm: Gray Level Size Zone Matrix (16 features)
    - ngtdm: Neighbouring Gray Tone Difference Matrix (5 features)
    
    Features are collected from every contour that is supplied along with the raw
    image. Depending on the amount of contours inside an image, this may result 
    in very large dataframes. 

    The specified channels correspond to the channels that can be selected in 
    phenopype.core.preprocessing.decompose_image. 


    Parameters
    ----------
    image : ndarray
        input image
    annotation: dict
        phenopype annotation containing contours
    features: ["firstorder", "shape", "glcm", "gldm", "glrlm", "glszm", "ngtdm"] list, optional
        type of texture features to extract
    channels : {"raw", "gray", "red", "green", "blue", "hue", "saturation", "value"}  str, optional
        image channel to extract texture features from
    min_diameter: int, optional
        minimum diameter of the contour (shouldn't be too small for sensible feature extraction')
        
    Returns
    -------
    annotations: dict
        phenopype annotation containing texture features

    """
    
    # =============================================================================
    # retain settings
    
    contour_id = kwargs.get("contour_id")
    
    ## retrieve settings from args
    local_settings  = utils_lowlevel._drop_dict_entries(
        locals(), drop=["image","kwargs", "annotation"])
        
    ## update settings from kwargs
    if kwargs:
        utils_lowlevel._update_settings(kwargs, local_settings)
        
    if not isinstance(channels, list):
        channels = [channels]

    feature_activation = {}
    for feature in features:
        feature_activation[feature] = []
        
        

	# =============================================================================
	# setup 
       
    contours = utils_lowlevel._provide_annotation_data(annotation, "contour", "coord_list", kwargs)
    contours_support = utils_lowlevel._provide_annotation_data(annotation, "contour", "support", kwargs)
    
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

        image_slice = preprocessing.decompose_image(image, channel)
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
