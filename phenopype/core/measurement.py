#%% modules
import cv2, copy
import numpy as np
import pandas as pd

import numpy.ma as ma

from phenopype.utils_lowlevel import (
    _image_viewer,
    _auto_line_width,
    _auto_point_size,
    _auto_text_width,
    _auto_text_size,
)

#%% methods


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
    Place landmarks.
    
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
