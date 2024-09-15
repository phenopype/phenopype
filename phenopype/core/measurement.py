#%% modules

import cv2
import math
import numpy as np
import sys
from tqdm import tqdm as _tqdm

from phenopype import __version__
from phenopype import _vars
from phenopype import decorators
from phenopype import utils_lowlevel as ul

#%% methods

@decorators.annotation_function
def set_landmark(
    image,
    point_colour="default",
    point_size="auto",
    label=True,
    label_colour="default",
    label_size="auto",
    label_width="auto",
    **kwargs,
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
    # annotation management

    fun_name = sys._getframe().f_code.co_name
    annotation_type = ul._get_annotation_type(fun_name)

    annotation = kwargs.get("annotation")

    gui_data = {_vars._coord_type: ul._get_GUI_data(annotation)}
    gui_settings = ul._get_GUI_settings(kwargs, annotation)

    # =============================================================================
    # execute

    gui = ul._GUI(
        image=image,
        tool="point",
        label=label,
        point_size=point_size,
        point_colour=point_colour,
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
            "point_size": point_size,
            "point_colour": point_colour,
            "label": label,
            "label_size": label_size,
            "label_width": label_width,
            "label_colour": label_colour,
        },
        "data": {annotation_type: gui.data[_vars._coord_type],},
    }

    if len(gui_settings) > 0:
        annotation["settings"]["GUI"] = gui_settings

    # =============================================================================
    # return

    return annotation

def set_polyline(
        image, 
        line_width="auto", 
        line_colour="default", 
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
    # execute

    gui = ul._GUI(
        image=image,
        tool="polyline",
        line_width=line_width,
        line_colour=line_colour,
        data=gui_data,
        **gui_settings,
    )

    line_coords = gui.data[_vars._coord_list_type]
    n_lines = len(line_coords)
    line_lengths = []
    for line in line_coords:
        line_lengths.append(round(ul._calc_distance_polyline(line), 1))
    
    # =============================================================================
    # assemble results
    
    annotation = {
        "info": {
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            "annotation_type": annotation_type,
        },
        "settings": {"line_width": line_width, "line_colour": line_colour,},
        "data": {
            "n": n_lines,
            "lengths": line_lengths,
            annotation_type: line_coords,
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



def detect_skeleton(annotations, thinning="zhangsuen", **kwargs):
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
    # annotation management

    ## get contours
    annotation_type = _vars._contour_type
    annotation_id = kwargs.get(annotation_type + "_id", None)
    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
    contours = annotation["data"][annotation_type]

    ##  features
    fun_name = sys._getframe().f_code.co_name
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    # =============================================================================
    # setup

    padding = kwargs.get("padding", 5)

    # =============================================================================
    # execute

    lines, line_lengths = [], []

    for coords in contours:
        rx, ry, rw, rh = cv2.boundingRect(coords)

        mask = np.zeros((rh + int(2 * padding), rw + int(2 * padding)), np.uint8)
        mask = cv2.fillPoly(mask, [coords], 255, offset=(-rx + padding, -ry + padding))

        skeleton = cv2.ximgproc.thinning(
            mask, thinningType=_vars.opencv_skeletonize_flags[thinning]
        )
        skel_contour, skel_hierarchy = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        skel_contour = skel_contour[0]
        skel_contour[:, :, 0] = skel_contour[:, :, 0] + rx - padding
        skel_contour[:, :, 1] = skel_contour[:, :, 1] + ry - padding
        skel_contour = ul._convert_arr_tup_list(skel_contour)[0]

        lines.append(skel_contour)
        line_lengths.append(round((ul._calc_distance_polyline(skel_contour)/2),1))
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
            "n": len(lines),
            "lengths": line_lengths,
            annotation_type: lines,
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


def compute_shape_features(annotations, features=["basic"], min_diameter=5, **kwargs):
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
        - roundness = (4 * contour_area) / (pi * contour_perimeter_length^2)
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
    # annotation management
    
    # print(annotations)

    ## get contours
    contour_id = kwargs.get(_vars._contour_type + "_id", None)
    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=_vars._contour_type,
        annotation_id=contour_id,
        kwargs=kwargs,
    )
    
    contours = annotation["data"][_vars._contour_type]
    contours_support = annotation["data"]["support"]
    
    ##  features
    fun_name = sys._getframe().f_code.co_name
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    # =============================================================================
    # execute

    shape_features = []

    for idx1, (coords, support) in enumerate(zip(contours, contours_support)):

        idx1 += 1

        output = {}

        if support["diameter"] > min_diameter:

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
                    "area": cnt_area,
                    "circularity": circularity,
                    "diameter": cnt_diameter,
                    "compactness": compactness,
                    "min_rect_max": min_rect_max,
                    "min_rect_min": min_rect_min,
                    "perimeter_length": perimeter_length,
                    "rect_height": rect_height,
                    "rect_width": rect_width,
                    "roundness": roundness,
                    "solidity": solidity,
                    "tri_area": tri_area,
                }

                output = {**output, **basic}

            ## moments
            if "moments" in features:
                moments = cv2.moments(coords)
                output = {**output, **moments}

            ## hu moments
            if "hu_moments" in features:
                if not "moments" in features:
                    moments = cv2.moments(coords)
                hu_moments = {}
                for idx2, mom in enumerate(cv2.HuMoments(moments)):
                    hu_moments["hu" + str(idx2 + 1)] = mom[0]
                output = {**output, **hu_moments}

            shape_features.append(output)

    # =============================================================================
    # return

    annotation = {
        "info": {
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            "annotation_type": annotation_type,
        },
        "settings": {
            "features": features,
            "min_diameter": min_diameter,
            "contour_id": contour_id,
            },
        "data": {annotation_type: shape_features},
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


def compute_texture_moments(
    image,
    annotations,
    features=["firstorder"],
    channel_names=["blue", "green", "red"],
    min_diameter=5,
    **kwargs,
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
    image, which, depending on the number of contours, may result in long computing 
    time and very large dataframes.

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
    channels : list, optional
        image channel to extract texture features from. if none is given, will extract from all channels in image
    min_diameter: int, optional
        minimum diameter of the contour (shouldn't be too small for sensible feature extraction')

    Returns
    -------
    annotations: dict
        phenopype annotation containing texture features

    """

    # =============================================================================
    # annotation management

    ## get contours
    contour_id = kwargs.get(_vars._contour_type + "_id", None)
    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=_vars._contour_type,
        annotation_id=contour_id,
        kwargs=kwargs,
    )
    contours = annotation["data"][_vars._contour_type]
    contours_support = annotation["data"]["support"]
    
    ##  features
    fun_name = sys._getframe().f_code.co_name
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    # =============================================================================
    # setup

    tqdm_off = kwargs.get("tqdm_off",True)

    feature_activation = {}
    for feature in features:
        feature_activation[feature] = []
        
    # =============================================================================
    # execute

    ## create forgeround mask
    foreground_mask_inverted = np.zeros(image.shape[:2], np.uint8)
    
    # print(contours)
    for coords in contours:
        foreground_mask_inverted = cv2.fillPoly(foreground_mask_inverted, [coords], 255)
        
    moments_features = []
    if image.ndim == 2:
        layers = 1
        image = np.dstack((image,image)) 
    else:
        layers = image.shape[2]
        
    if len(channel_names) > layers:
        print("- Warning: more channels provided than in given image - skipping excess ones!")
    
    for idx1, (coords, support) in _tqdm(
            enumerate(zip(contours, contours_support)),
            "Computing basic moments",
            total=len(contours),
            disable=tqdm_off
    ):

        output = {}
        
        if support["diameter"] > min_diameter:

            for idx2, channel in enumerate(channel_names):

                if (idx2 + 1) > image.shape[2]:
                    continue
                
                rx, ry, rw, rh = cv2.boundingRect(coords)
                data = image[ry : ry + rh, rx : rx + rw, idx2]
                mask = foreground_mask_inverted[ry : ry + rh, rx : rx + rw]            
                
                if len(np.unique(mask)) > 1:
                    
                    # Apply mask
                    masked_data = data[mask != 0]

                    # Compute first-order statistics
                    mean = np.mean(masked_data)
                    std = np.std(masked_data)
                    var = np.var(masked_data)
                    skewness = np.mean((masked_data - mean)**3) / (std**3)
                    kurtosis = np.mean((masked_data - mean)**4) / (std**4) - 3

                    output[channel + "_mean"] = float(mean)
                    output[channel + "_std"] = float(std)
                    output[channel + "_variance"] = float(var)
                    output[channel + "_skewness"] = float(skewness)
                    output[channel + "_kurtosis"] = float(kurtosis)

                else:
                    continue

        moments_features.append(output)
       
        

    # =============================================================================
    # return

    annotation = {
        "info": {
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            "annotation_type": annotation_type,
        },
        "settings": {
            "features": features,
            "min_diameter": min_diameter,
            "channels_names": channel_names,
            "contour_id": contour_id,
        },
        "data": {annotation_type: moments_features},
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
