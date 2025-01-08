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
from phenopype import core

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


def compute_shape_moments(annotations, features=["basic"], min_diameter=5, **kwargs):
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
                    "shape_stats_area": cnt_area,
                    "shape_stats_circularity": circularity,
                    "shape_stats_diameter": cnt_diameter,
                    "shape_stats_compactness": compactness,
                    "shape_stats_min_rect_max": min_rect_max,
                    "shape_stats_min_rect_min": min_rect_min,
                    "shape_stats_perimeter_length": perimeter_length,
                    "shape_stats_rect_height": rect_height,
                    "shape_stats_rect_width": rect_width,
                    "shape_stats_roundness": roundness,
                    "shape_stats_solidity": solidity,
                    "shape_stats_tri_area": tri_area,
                }

                output = {**output, **basic}

            ## moments
            if "moments" in features:
                moments = cv2.moments(coords)
                moments_save = {f"shape_moments_{key}": value for key, value in moments.items()}
                output = {**output, **moments_save}

            ## hu moments
            if "hu_moments" in features:
                if not "moments" in features:
                    moments = cv2.moments(coords)
                hu_moments = {}
                for idx2, mom in enumerate(cv2.HuMoments(moments)):
                    hu_moments["hu" + str(idx2 + 1)] = mom[0]
                hu_moments_save = {f"shape_moments_{key}": value for key, value in hu_moments.items()}
                output = {**output, **hu_moments_save}

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


def compute_color_moments(
    image,
    annotations,
    channel_names=["blue", "green", "red"],
    min_diameter=5,
    **kwargs,
):
    """
    Compute statistical color moments for regions of interest (ROIs) in an image.

    This function calculates the mean, standard deviation, variance, skewness, 
    and kurtosis for pixel intensities within each specified ROI and image channel.
    The results are stored in the phenopype `annotations` dictionary.

    Parameters
    ----------
    image : ndarray
        Input image as a NumPy array. Can have multiple channels (e.g., RGB).
    annotations : dict
        Phenopype annotation dictionary containing contours and support information 
        for regions of interest (ROIs).
    channel_names : list, optional
        Names of the image channels to extract color moments from. If not provided, 
        the default is ["blue", "green", "red"].
    min_diameter : int, optional
        Minimum allowable diameter for contours to be included in the analysis.
        ROIs smaller than this size are excluded to ensure meaningful feature extraction.
    **kwargs : dict, optional
        Additional arguments for configuration, including:
        - `"tqdm_off"`: Disable progress bar if set to `True` (default).
        - `"annotation_id"`: Specific annotation ID to process.

    Returns
    -------
    annotations : dict
        Updated phenopype annotation dictionary with computed color moments for each ROI:
        - **First-order moments**:
          - `color_moments_{channel_name}_mean`: Mean pixel intensity.
          - `color_moments_{channel_name}_var`: Variance of pixel intensity.
          - `color_moments_{channel_name}_skew`: Skewness of the intensity distribution.
          - `color_moments_{channel_name}_kurt`: Kurtosis of the intensity distribution.
        - **Dispersion and variability**:
          - `color_moments_{channel_name}_median`: Median pixel intensity.
          - `color_moments_{channel_name}_iqr`: Interquartile range (P75 - P25).
          - `color_moments_{channel_name}_rmad`: Robust mean absolute deviation, based on the 10th to 90th percentile range.
          - `color_moments_{channel_name}_entropy`: Entropy of the pixel intensity distribution, indicating uncertainty or randomness.
          - `color_moments_{channel_name}_uniformity`: Uniformity of the pixel intensity distribution, measuring homogeneity.
          - `color_moments_{channel_name}_cv`: Coefficient of variation (normalized standard deviation as a ratio of the mean).
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

    ## progress bar management
    tqdm_off = kwargs.get("tqdm_off",True)
    
    ## channel formatting
    if not isinstance(channel_names, list):
        channel_names = [channel_names]
        
    ## create stack that works with 2D or ND imgs
    channels = cv2.split(image)
        
    # =============================================================================
    # execute
    
    results = []
    for idx1, (coords, support) in _tqdm(
            enumerate(zip(contours, contours_support)),
            "Computing color moments",
            total=len(contours),
            disable=tqdm_off
    ):

        
        ## get roi mask
        rx, ry, rw, rh = cv2.boundingRect(coords)
        roi_mask = np.zeros((rh, rw), np.uint8)
        roi_mask = cv2.fillPoly(roi_mask, [coords], 255, 0, 0, (rx, ry))
        
        ## go through channels locally
        output = {}
        if support["diameter"] > min_diameter and len(np.unique(roi_mask)) > 1:
            for channel_name, channel in zip(channel_names, channels):
                
                ## get roi
                roi = channel[ry : ry + rh, rx : rx + rw]
                masked_data = roi[roi_mask != 0]

                ## computations
                mean = float(np.mean(masked_data))
                std = float(np.std(masked_data))
                var = float(np.var(masked_data)) #std**2
                epsilon = np.finfo(float).eps
                unique_values, counts = np.unique(masked_data, return_counts=True)
                probabilities = counts / counts.sum()
                p25 = np.percentile(masked_data, 25)
                p75 = np.percentile(masked_data, 75)
                lower_bound = np.percentile(masked_data, 10)
                upper_bound = np.percentile(masked_data, 90)
                data_within_bounds = masked_data[(masked_data >= lower_bound) & (masked_data <= upper_bound)]
                mean_within_bounds = np.mean(data_within_bounds)

                # firstorder moments
                output[f"color_moments_{channel_name}_mean"] = mean
                output[f"color_moments_{channel_name}_var"] = var
                output[f"color_moments_{channel_name}_skew"] = float(np.mean((masked_data - mean)**3) / (std**3))
                output[f"color_moments_{channel_name}_kurt"] = float(np.mean((masked_data - mean)**4) / (std**4) - 3)
                
                # dispersion and variability
                output[f"color_moments_{channel_name}_median"] = float(np.median(masked_data))
                output[f"color_moments_{channel_name}_std"] = std
                output[f"color_moments_{channel_name}_iqr"] = float(p75 - p25)
                output[f"color_moments_{channel_name}_rmad"] = float(np.mean(np.abs(data_within_bounds - mean_within_bounds)))
                output[f"color_moments_{channel_name}_entropy"] = -float(np.sum(probabilities * np.log2(probabilities + epsilon)))
                output[f"color_moments_{channel_name}_uniformity"] = float(np.sum(probabilities**2))
                output[f"color_moments_{channel_name}_cv"] = float(std / mean) if mean != 0 else 0

        results.append(output)
       
    # =============================================================================
    # return

    annotation = {
        "info": {
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            "annotation_type": annotation_type,
        },
        "settings": {
            "min_diameter": min_diameter,
            "channels_names": channel_names,
            "contour_id": contour_id,
        },
        "data": {annotation_type: results},
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

def compute_DFT_stats(image, col_space="bgr", channel="gray", resize=None):
    
    """
    Compute Discrete Fourier Transform (DFT) statistics for an image.
    
    This function performs the following steps:
    1. Converts the image to the specified color space and extracts the selected channel.
    2. Optionally resizes the image to a square of size `resize x resize`.
    3. Applies the DFT to compute the magnitude spectrum of the frequency domain.
    4. Calculates regional statistics on a 3x3 grid of the magnitude spectrum.
        (see https://dsp.stackexchange.com/q/95090/46324)
    5. Computes global statistics on the entire magnitude spectrum, including mean, standard deviation,
       and a complexity index based on the proportion of high-frequency components.
    
    Parameters:
    ----------
    image : np.ndarray
        Input image as a NumPy array.
    col_space : str, optional
        Color space to convert the image to (default is "bgr").
    channel : str, optional
        Specific channel to extract for analysis (default is "gray").
    resize : int, optional
        If specified, resizes the image to `resize x resize` dimensions (default is None).
    
    Returns:
    -------
    results : dict
        A dictionary containing the following keys:
        - `color_dft_{channel}_dia`: Mean intensity of diagonal regions in the 3x3 grid.
        - `color_dft_{channel}_ver`: Mean intensity of vertical edge regions in the 3x3 grid.
        - `color_dft_{channel}_hor`: Mean intensity of horizontal edge regions in the 3x3 grid.
        - `color_dft_{channel}_low`: Mean intensity of the central region in the 3x3 grid.
        - `color_dft_{channel}_meanmag`: Mean value of the entire magnitude spectrum.
        - `color_dft_{channel}_stdmag`: Standard deviation of the entire magnitude spectrum.
        - `color_dft_{channel}_hifreq`: Proportion of high-frequency components, normalized by image size.
    """
        
    ## slice and resize 
    img2d = core.preprocessing.decompose_image(image, col_space=col_space, channels=channel)
    if resize:
        height, width = resize, resize
        img2d = cv2.resize(img2d, (height, width))
    else:
        height, width = image.shape[:2]
    
    # apply the DFT (Discrete Fourier Transform)
    dft = cv2.dft(np.float32(img2d), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

    ## stats on regions of magnitude spectrum
    h_step, w_step, means = height // 3, width // 3, []
    for i in range(3):
        for j in range(3):
            region = magnitude_spectrum[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            means.append(np.mean(region))
    results = {
        f"color_dft_{channel}_dia": np.mean([means[0], means[2], means[6], means[8]]),
        f"color_dft_{channel}_ver": np.mean([means[1], means[7]]),
        f"color_dft_{channel}_hor": np.mean([means[3], means[5]]),
        f"color_dft_{channel}_low": means[4]
        }
    
    ## stats on the entire magnitude spectrum  
    mean_magnitude = np.mean(magnitude_spectrum)
    std_magnitude = np.std(magnitude_spectrum)
    high_freq_count = np.sum(magnitude_spectrum > (mean_magnitude + std_magnitude))
    complexity_index = high_freq_count / (height * width)
    results.update({
        f"color_dft_{channel}_meanmag": mean_magnitude,
        f"color_dft_{channel}_stdmag": std_magnitude,
        f"color_dft_{channel}_hifreq": complexity_index,  
    })
    
    return results
