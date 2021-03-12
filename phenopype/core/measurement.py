#%% modules
import cv2, copy
import math
import numpy as np
import numpy.ma as ma
import pandas as pd

import logging
from radiomics import featureextractor
import SimpleITK as sitk
from tqdm import tqdm

from phenopype.utils_lowlevel import (
    _image_viewer,
    _auto_line_width,
    _auto_point_size,
    _auto_text_width,
    _auto_text_size,
)

#%% methods


def colour_intensity(
    obj_input,
    df_image_data=None,
    df_contours=None,
    channels="gray",
    background=False,
    background_ext=20,
):
    """
    Measures pixel values within the provided contours, either across all 
    channels ("gray") or for each channel separately ("rgb"). Measures mean 
    and standard deviation
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
    Returns
    -------
    df_colours : DataFrame or container
        contains the pixel intensities 
    """
    ## kwargs
    if channels.__class__.__name__ == "str":
        channels = [channels]
    if background == True:
        flag_background = background
        q = background_ext
    else:
        flag_background = False

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename": "unknown"}, index=[0])
        if df_contours.__class__.__name__ == "NoneType":
            print("no df supplied - cannot measure colour intensity")
            return
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image_copy)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_contours"):
            df_contours = obj_input.df_contours
    else:
        print("wrong input format.")
        return

    ## make dataframe backbone
    df_colours = pd.DataFrame(df_contours["contour"])

    ## create forgeround mask
    foreground_mask_inverted = np.zeros(image.shape[:2], np.uint8)
    for index, row in df_contours.iterrows():
        foreground_mask_inverted = cv2.fillPoly(
            foreground_mask_inverted, [row["coords"]], 255
        )
    foreground_mask = np.invert(np.array(foreground_mask_inverted))

    ## grayscale
    if "gray" in channels:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_cols = {"gray_mean": "NA", "gray_sd": "NA"}
        df_colours = df_colours.assign(**new_cols)
        for index, row in df_contours.iterrows():
            rx, ry, rw, rh = cv2.boundingRect(row["coords"])
            grayscale = ma.array(
                data=image_gray[ry : ry + rh, rx : rx + rw],
                mask=foreground_mask[ry : ry + rh, rx : rx + rw],
            )
            df_colours.at[index, ["gray_mean", "gray_sd"]] = (
                np.ma.mean(grayscale),
                np.ma.std(grayscale),
            )

    ## red, green, blue
    if "rgb" in channels:
        df_colours = df_colours.assign(
            **{
                "red_mean": "NA",
                "red_sd": "NA",
                "green_mean": "NA",
                "green_sd": "NA",
                "blue_mean": "NA",
                "blue_sd": "NA",
            }
        )
        for index, row in df_contours.iterrows():
            rx, ry, rw, rh = cv2.boundingRect(row["coords"])
            blue = ma.array(
                data=image[ry : ry + rh, rx : rx + rw, 0],
                mask=foreground_mask[ry : ry + rh, rx : rx + rw],
            )
            green = ma.array(
                data=image[ry : ry + rh, rx : rx + rw, 1],
                mask=foreground_mask[ry : ry + rh, rx : rx + rw],
            )
            red = ma.array(
                data=image[ry : ry + rh, rx : rx + rw, 2],
                mask=foreground_mask[ry : ry + rh, rx : rx + rw],
            )
            df_colours.at[index, ["red_mean", "red_sd"]] = np.ma.mean(red), np.ma.std(red)
            df_colours.at[index, ["green_mean", "green_sd"]] = (
                np.ma.mean(green),
                np.ma.std(green),
            )
            df_colours.at[index, ["blue_mean", "blue_sd"]] = (
                np.ma.mean(blue),
                np.ma.std(blue),
            )

    ## background grayscale
    if flag_background:
        df_colours = df_colours.assign(**{"gray_mean_b": "NA", "gray_sd_b": "NA"})
        for index, row in df_contours.iterrows():
            rx, ry, rw, rh = cv2.boundingRect(row["coords"])
            foreground_mask_inverted_extended = foreground_mask_inverted[
                (ry - q) : (ry + rh + q), (rx - q) : (rx + rw + q)
            ]
            grayscale = ma.array(
                data=image_gray[(ry - q) : (ry + rh + q), (rx - q) : (rx + rw + q)],
                mask=foreground_mask_inverted_extended,
            )
            df_colours.at[index, ["gray_mean_b", "gray_sd_b"]] = (
                np.ma.mean(grayscale),
                np.ma.std(grayscale),
            )

    ## merge with existing image_data frame
    df_colours = pd.concat(
        [
            pd.concat([df_image_data] * len(df_colours)).reset_index(drop=True),
            df_colours.reset_index(drop=True),
        ],
        axis=1,
    )

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return df_colours
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_colours = df_colours

def landmarks(
    obj_input,
    df_image_data=None,
    overwrite=False,
    point_colour="green",
    point_size="auto",
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
    obj_input : array or container
        input object
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata, will be added to landmark
        output DataFrame
    overwrite: bool, optional
        if working using a container, or from a phenopype project directory, 
        should existing landmarks be overwritten
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

    ## kwargs
    flag_overwrite = overwrite
    test_params = kwargs.get("test_params", {})

    ## load image
    df_landmarks = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename": "unknown"}, index=[0])
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image_copy)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_landmarks"):
            df_landmarks = obj_input.df_landmarks
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if point_size == "auto":
        point_size = _auto_point_size(image)
    if label_size == "auto":
        label_size = _auto_text_size(image)
    if label_width == "auto":
        label_width = _auto_text_width(image)

    while True:
        ## check if exists
        if not df_landmarks.__class__.__name__ == "NoneType" and flag_overwrite == False:
            df_landmarks = df_landmarks[
                df_landmarks.columns.intersection(["landmark", "x", "y"])
            ]
            print("- landmarks already set (overwrite=False)")
            break
        elif not df_landmarks.__class__.__name__ == "NoneType" and flag_overwrite == True:
            print("- setting landmarks (overwriting)")
            pass
        elif df_landmarks.__class__.__name__ == "NoneType":
            print("- setting landmarks")
            pass

        ## set landmarks
        out = _image_viewer(
            image,
            tool="landmarks",
            point_size=point_size,
            point_colour=point_colour,
            label_size=label_size,
            label_width=label_width,
            label_colour=label_colour,
            previous=test_params,
        )
        coords = out.points

        ## abort
        if not out.done:
            if obj_input.__class__.__name__ == "ndarray":
                print("terminated polyline creation")
                return
            elif obj_input.__class__.__name__ == "container":
                print("- terminated polyline creation")
                return True

        ## make df
        df_landmarks = pd.DataFrame(coords, columns=["x", "y"])
        df_landmarks.reset_index(inplace=True)
        df_landmarks.rename(columns={"index": "landmark"}, inplace=True)
        df_landmarks["landmark"] = df_landmarks["landmark"] + 1
        break

    ## merge with existing image_data frame
    df_landmarks = pd.concat(
        [
            pd.concat([df_image_data] * len(df_landmarks)).reset_index(drop=True),
            df_landmarks.reset_index(drop=True),
        ],
        axis=1,
    )

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return df_landmarks
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_landmarks = df_landmarks




def polylines(
    obj_input,
    df_image_data=None,
    overwrite=False,
    line_width="auto",
    line_colour="blue",
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
    ## kwargs
    flag_overwrite = overwrite

    ## load image
    df_polylines = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename": "unknown"}, index=[0])
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image_copy)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_polylines"):
            df_polylines = obj_input.df_polylines
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if line_width == "auto":
        line_width = _auto_line_width(image)
    test_params = kwargs.get("test_params", {})

    while True:
        ## check if exists
        if not df_polylines.__class__.__name__ == "NoneType" and flag_overwrite == False:
            df_polylines = df_polylines[
                df_polylines.columns.intersection(["polyline", "length", "x", "y"])
            ]
            print("- polylines already drawn (overwrite=False)")
            break
        elif not df_polylines.__class__.__name__ == "NoneType" and flag_overwrite == True:
            print("- draw polylines (overwriting)")
            pass
        elif df_polylines.__class__.__name__ == "NoneType":
            print("- draw polylines")
            pass

        ## method
        out = _image_viewer(
            image,
            tool="polyline",
            line_width=line_width,
            line_colour=line_colour,
            previous=test_params,
        )
        coords = out.point_list

        ## abort
        if not out.done:
            if obj_input.__class__.__name__ == "ndarray":
                print("terminated polyline creation")
                return
            elif obj_input.__class__.__name__ == "container":
                print("- terminated polyline creation")
                return True

        ## create df
        df_polylines = pd.DataFrame(columns=["polyline", "length", "x", "y"])
        idx = 0
        for point_list in coords:
            idx += 1
            arc_length = int(cv2.arcLength(np.array(point_list), closed=False))
            df_sub = pd.DataFrame(point_list, columns=["x", "y"])
            df_sub["polyline"] = idx
            df_sub["length"] = arc_length
            df_polylines = df_polylines.append(df_sub, ignore_index=True, sort=False)
        break

    ## merge with existing image_data frame
    df_polylines = pd.concat(
        [
            pd.concat([df_image_data] * len(df_polylines)).reset_index(drop=True),
            df_polylines.reset_index(drop=True),
        ],
        axis=1,
    )

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return df_polylines
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_polylines = df_polylines


def skeletonize(
    obj_input, df_image_data=None, df_contours=None, thinning="zhangsuen", padding=10
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

    ## kwargs
    skeleton_alg = {
        "zhangsuen": cv2.ximgproc.THINNING_ZHANGSUEN,
        "guohall": cv2.ximgproc.THINNING_GUOHALL,
    }

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename": "unknown"}, index=[0])
        if df_contours.__class__.__name__ == "NoneType":
            print("no df supplied - cannot measure colour intensity")
            return
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image_copy)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_contours"):
            df_contours = obj_input.df_contours
    else:
        print("wrong input format.")
        return

    ## create forgeround mask
    df_contours = df_contours.assign(
        **{"skeleton_perimeter": "NA", "skeleton_coords": "NA"}
    )

    for index, row in df_contours.iterrows():
        coords = copy.deepcopy(row["coords"])
        rx, ry, rw, rh = cv2.boundingRect(coords)
        image_sub = image[
            (ry - padding) : (ry + rh + padding), (rx - padding) : (rx + rw + padding)
        ]

        mask = np.zeros(image_sub.shape[0:2], np.uint8)
        mask = cv2.fillPoly(mask, [coords], 255, offset=(-rx + padding, -ry + padding))

        skeleton = cv2.ximgproc.thinning(mask, thinningType=skeleton_alg[thinning])
        skel_ret, skel_contour, skel_hierarchy = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        perimeter = int(cv2.arcLength(skel_contour[0], closed=False) / 2)

        skel_contour = skel_contour[0]
        skel_contour[:, :, 0] = skel_contour[:, :, 0] + rx - padding
        skel_contour[:, :, 1] = skel_contour[:, :, 1] + ry - padding

        df_contours.at[index, ["skeleton_perimeter", "skeleton_coords"]] = (
            perimeter,
            skel_contour,
        )

    if obj_input.__class__.__name__ == "ndarray":
        return df_contours
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_contours = df_contours


def shape_features(
    obj_input,
    df_contours=None,
    resize=True,
    resize_to=100,
    return_basic=True,
    return_moments=False, 
    return_hu_moments=True,

):
    """
    Collects a set of 41 shape descriptors from every contour. There are three sets of 
    descriptors: basic shape descriptors, moments, and hu moments. Two additional features,
    contour area and diameter are already provided by the find_contours function.
    https://docs.opencv.org/3.4.9/d3/dc0/group__imgproc__shape.html

    Of the basic shape descriptors, all 10 are translational invariants, 8 are rotation 
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

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df_contours = obj_input
    elif obj_input.__class__.__name__ == "container":
        if hasattr(obj_input, "df_contours"):
            df_contours = obj_input.df_contours
    else:
        print("wrong input format.")
        return
    
    if df_contours.__class__.__name__ == "NoneType":
        print("no df supplied - cannot measure shape features")
        return
    
    ## make dataframe backbone
    df_shapes = copy.deepcopy(df_contours.drop(columns=["coords"]))

    ## custom shape descriptors
    desc_basic_shape = ['circularity',
                         'compactness',
                         'min_rect_max',
                         'min_rect_min',
                         'perimeter_length',
                         'rect_height',
                         'rect_width',
                         'roundness',
                         'solidity',
                         'tri_area']
    for name in desc_basic_shape:
        df_shapes = df_shapes.assign(**{name: "NA"})

    ## moments 
    desc_moments = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 
                'm12', 'm03', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12',
                'mu03', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
    for name in desc_moments:
        df_shapes = df_shapes.assign(**{name: "NA"})

    ## hu moments
    desc_hu = ['hu1','hu2','hu3','hu4','hu5','hu6','hu7']
    for name in desc_hu:
        df_shapes = df_shapes.assign(**{name: "NA"})


    ## calculate shape descriptors from contours
    for index, row in df_contours.iterrows():
        
        ## contour coords
        coords = row["coords"]
        
        if resize==True and "current_px_mm_ratio" in row:
            factor = resize_to / row["current_px_mm_ratio"]        
            coords_norm = coords - row["center"]
            coords_scaled = coords_norm * factor
            coords = coords_scaled + row["center"]
            coords = coords.astype(np.int32)
            
            ## correct in df
            df_shapes.loc[index, "diameter"] = int(row["diameter"] * factor)
            df_shapes.loc[index, "area"]  = int(cv2.contourArea(coords))
            
        ## retrieve area and diameter
        cnt_diameter = df_contours.loc[index]["diameter"]
        cnt_area = df_contours.loc[index]["area"]

        ## custom shape descriptors
        convex_hull = cv2.convexHull(coords)
        tri_area, tri_coords = cv2.minEnclosingTriangle(coords)
        min_rect_center, min_rect_min_max, min_rect_angle = cv2.minAreaRect(coords)
        min_rect_min, min_rect_max = min_rect_min_max[0], min_rect_min_max[1]
        rect_x, rect_y, rect_width, rect_height = cv2.boundingRect(coords)
        perimeter_length = cv2.arcLength(coords, closed=True)
        circularity = 4 * np.pi * cnt_area / math.pow(perimeter_length, 2)
        roundness = (4 * cnt_area) / (np.pi * math.pow(cnt_diameter, 2))
        solidity = cnt_area / cv2.contourArea(convex_hull)
        compactness = math.sqrt(4 * cnt_area / np.pi) / cnt_diameter
        
        df_shapes.at[index, desc_basic_shape] = (
            circularity,
            compactness,
            min_rect_max,
            min_rect_min,
            perimeter_length,
            rect_height,
            rect_width,
            roundness,
            solidity,
            tri_area
        )    
                     
        ## moments
        moments = cv2.moments(coords)
        df_shapes.at[index, desc_moments] = list(moments.values())
                     
        ## hu moments
        hu_moments = cv2.HuMoments(moments)
        hu_moments_list = []
        for i in hu_moments:
            hu_moments_list.append(i[0])
        df_shapes.at[index, desc_hu] = hu_moments_list
        
        
    ## drop unwanted columns
    if return_basic == False:
        df_shapes.drop(desc_basic_shape, axis=1, inplace=True)
    if return_moments == False:
        df_shapes.drop(desc_moments, axis=1, inplace=True)
    if return_hu_moments == False:
        df_shapes.drop(desc_hu, axis=1, inplace=True)
        
    ## return
    if obj_input.__class__.__name__ == "DataFrame":
        return df_shapes
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_shapes = df_shapes



def texture_features(
    obj_input,
    df_image_data=None,
    df_contours=None,
    channels="gray",
    background=False,
    background_ext=20,
    min_diameter=5,
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

    Returns
    -------
    df_textures : DataFrame or container
        contains the pixel intensities 

    """
    ## kwargs
    if channels.__class__.__name__ == "str":
        channels = [channels]
    if background == True:
        flag_background = background
        q = background_ext
    else:
        flag_background = False

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename": "unknown"}, index=[0])
        if df_contours.__class__.__name__ == "NoneType":
            print("no df supplied - cannot measure colour intensity")
            return
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image_copy)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_contours"):
            df_contours = obj_input.df_contours
    else:
        print("wrong input format.")
        return
    
    ## deactivate warnigns from pyradiomics feature extractor
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    ## make dataframe backbone
    df_textures_meta = df_contours.drop(columns=["coords"])

    ## create forgeround mask
    foreground_mask_inverted = np.zeros(image.shape[:2], np.uint8)
    for index, row in df_contours.iterrows():
        foreground_mask_inverted = cv2.fillPoly(
            foreground_mask_inverted, [row["coords"]], 255
        )

    ## method
    df_df_list = []
    for channel in channels:
        df_textures_list = []

        if channel in ["grey","gray", "grayscale"]:
            channel = "gray"
            if len(image.shape)==3:
                image_data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_data = copy.deepcopy(image)
        elif channel in ["r","red"]:
            channel = "red"
            image_data = image[:,:,0]
        elif channel in ["g","green"]:
            channel = "green"
            image_data = image[:,:,1]
        elif channel in ["b","blue"]:
            channel = "blue"
            image_data = image[:,:,2]
        for index, row in tqdm(df_contours.iterrows(), 
                               desc="Processing " + channel + " texture features", 
                               total=df_contours.shape[0]):
            if row["diameter"] > min_diameter:
                rx, ry, rw, rh = cv2.boundingRect(row["coords"])

                data=image_data[ry : ry + rh, rx : rx + rw]
                mask=foreground_mask_inverted[ry : ry + rh, rx : rx + rw]
                sitk_data = sitk.GetImageFromArray(data)
                sitk_mask = sitk.GetImageFromArray(mask)
                
                # pp.show_image(foreground_mask_inverted)

                extractor = featureextractor.RadiomicsFeatureExtractor()
                features = extractor.execute(sitk_data, sitk_mask, label=255)
                
                output = {}
                for key, val in features.items():
                    if not "diagnostics" in key :
                        output[key.split('_', 1)[1]  ] = val
                df_textures_list.append(output)
            else:
                df_textures_list.append({})
                
        df_textures_int = pd.DataFrame(df_textures_list)
        df_textures_int.insert(0, column="Channel", value=channel)
        df_df_list.append(pd.concat([df_textures_meta, df_textures_int], axis=1))
        
    df_textures = pd.concat(df_df_list)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return df_textures
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_textures = df_textures

