#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import numpy.ma as ma
import pandas as pd

from datetime import datetime

from phenopype.utils_lowlevel import _image_viewer, _auto_line_width, _auto_point_size, _auto_text_width, _auto_text_size
from phenopype.settings import colours

#%% methods

def landmarks(obj_input, **kwargs):

    """Set landmarks, with option to measure length and enter specimen ID.
    
    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    scale: num (default: 1)
        pixel to mm-ratio 
    ID: str (default: NA)
        specimen ID; "query" is special flag for user entry
    draw_line: bool (default: False)
        flag to draw arc and measure it's length
    zoom_factor: int (default 5)
        magnification factor on mousewheel use
    show: bool (default: False)
        display the set landmarks 
    point_size: num (default: 1/300 of image diameter)
        size of the landmarks on the image in pixels
    point_col: value (default: red)
        colour of landmark (red, green, blue, black, white)
    label_size: num (default: 1/1500 of image diameter)
        size of the numeric landmark label in pixels
    label_col: value (default: black)
        colour of label (red, green, blue, black, white)    

    Returns
    -------
    .df = pandas data frame with landmarks (and arc-length, if selected)
    .drawn = image array with drawn landmarks (and lines)
    .ID = provided specimen ID 
    """
    
    ## kwargs
    df_image_data = kwargs.get("df_image_data", None)
    flag_overwrite = kwargs.get("overwrite", False)

    ## load image
    df_landmarks = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"})
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_landmarks"):
            df_landmarks = obj_input.df_landmarks
    else:
        warnings.warn("wrong input format.")
        return
    
    ## only landmark df
    df_landmarks = df_landmarks[df_landmarks.columns.intersection(['landmark','x','y'])]

    ## more kwargs
    point_size = kwargs.get("point_size", _auto_point_size(image))
    point_col = kwargs.get("point_col", "red")
    label_size = kwargs.get("label_size", _auto_text_size(image))
    label_width = kwargs.get("label_width", _auto_text_width(image))
    label_col = kwargs.get("label_col", "black")

    while True:
        ## check if exists
        if not df_landmarks.__class__.__name__ == "NoneType" and flag_overwrite == False:
            print("- landmarks already set (overwrite=False)")
            break
        elif not df_landmarks.__class__.__name__ == "NoneType" and flag_overwrite == True:
            print("- set landmarks (overwriting)")
            pass
        elif df_landmarks.__class__.__name__ == "NoneType":
            print("- set landmarks")
            pass
        
        ## set landmarks
        out = _image_viewer(image, tool="landmarks", 
                            point_size=point_size, 
                            point_col=point_col, 
                            label_size=label_size,
                            label_width=label_width, 
                            label_col=label_col)
        coords = out.points
        
        ## abort
        if not out.done:
            if obj_input.__class__.__name__ == "ndarray":
                warnings.warn("terminated polyline creation")
                return 
            elif obj_input.__class__.__name__ == "container":
                print("- terminated polyline creation")
                return True
    
        ## make df
        df_landmarks = pd.DataFrame(coords, columns=["x","y"])
        df_landmarks.reset_index(inplace=True)
        df_landmarks.rename(columns={"index": "landmark"},inplace=True)
        df_landmarks["landmark"] = df_landmarks["landmark"] + 1
        break

    ## merge with existing image_data frame
    df_landmarks = pd.concat([pd.concat([df_image_data]*len(df_landmarks)).reset_index(drop=True), 
                            df_landmarks.reset_index(drop=True)], axis=1)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
            return image, df_landmarks
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_landmarks = df_landmarks
        obj_input.canvas = image


def polylines(obj_input, **kwargs):
    """
    

    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ## kwargs
    df_image_data = kwargs.get("df_image_data", None)
    flag_overwrite = kwargs.get("overwrite", False)

    ## load image
    df_polylines = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"})
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_polylines"):
            df_polylines = obj_input.df_polylines
    else:
        warnings.warn("wrong input format.")
        return

    ## more kwargs
    line_width = kwargs.get("line_width", _auto_line_width(image))

    ## check if exists
    if not df_polylines.__class__.__name__ == "NoneType" and flag_overwrite == False:
        print("- polylines already drawn (overwrite=False)")
        return
    elif not df_polylines.__class__.__name__ == "NoneType" and flag_overwrite == True:
        print("- draw polylines (overwriting)")
    elif df_polylines.__class__.__name__ == "NoneType":
        print("- draw polylines")
        
    ## method
    out = _image_viewer(image, tool="polyline")
    coords = out.point_list
    
    ## abort
    if not out.done:
        if obj_input.__class__.__name__ == "ndarray":
            warnings.warn("terminated polyline creation")
            return 
        elif obj_input.__class__.__name__ == "container":
            print("- terminated polyline creation")
            return True

    ## create df
    df_polylines = pd.DataFrame(columns=["polyline", "length", "x", "y"])
    idx = 0
    for point_list in out.point_list:
        idx += 1
        arc_length = int(cv2.arcLength(np.array(point_list), closed=False))
        df_sub = pd.DataFrame(point_list, columns=["x","y"])
        df_sub["polyline"] = idx
        df_sub["length"] = arc_length
        df_polylines = df_polylines.append(df_sub, ignore_index=True, sort=False)
    df_polylines = pd.concat([pd.concat([df_image_data]*len(df_polylines)).reset_index(drop=True), 
                            df_polylines.reset_index(drop=True)], axis=1)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
            return image, df_polylines
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_polylines = df_polylines
        obj_input.canvas = image



def colour(obj_input, **kwargs):

    ## kwargs
    channels = kwargs.get("channels", ["gray"])
    df_contours = kwargs.get("df_contours", None)

    ## load image
    df_contours = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"})
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_contours"):
            df_contours = obj_input.df_contours
    else:
        warnings.warn("wrong input format.")
        return

    ## create forgeround mask
    image_bin = np.zeros(image.shape[:2], np.uint8)
    for index, row in df_contours.iterrows():
        if row["order"]=="parent":
            image_bin = cv2.fillPoly(image_bin, [row["coords"]], 255)
        elif row["order"]=="child":
            image_bin = cv2.fillPoly(image_bin, [row["coords"]], 0)
    foreground_mask = np.invert(np.array(image_bin, dtype=np.bool))

    ## make df
    df_colours = df_contours.drop(columns=['center','diameter','area','order','idx_child','idx_parent','coords'])
    
    ## grayscale
    if "gray" in channels:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_cols = {"gray_mean":"NA",
                    "gray_sd":"NA"}
        df_colours = df_colours.assign(**new_cols)
        for index, row in df_contours.iterrows():
            rx,ry,rw,rh = cv2.boundingRect(row["coords"])
            grayscale =  ma.array(data=image_gray[ry:ry+rh,rx:rx+rw], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            df_colours.at[index, ["gray_mean","gray_sd"]] = np.ma.mean(grayscale), np.ma.std(grayscale)

    ## red, green, blue
    if "rgb" in channels:
        df_colours = df_colours.assign(**{"red_mean":"NA",
                           "red_sd":"NA",
                           "green_mean":"NA",
                           "green_sd":"NA",
                           "blue_mean":"NA",
                           "blue_sd":"NA"})
        for index, row in df_contours.iterrows():
            rx,ry,rw,rh = cv2.boundingRect(row["coords"])
            blue =  ma.array(data=image[ry:ry+rh,rx:rx+rw,0], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            green =  ma.array(data=image[ry:ry+rh,rx:rx+rw,1], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            red =  ma.array(data=image[ry:ry+rh,rx:rx+rw,2], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            df_colours.at[index, ["red_mean","red_sd"]]  = np.ma.mean(red), np.ma.std(red)
            df_colours.at[index, ["green_mean","green_sd"]]  = np.ma.mean(green), np.ma.std(green)
            df_colours.at[index, ["blue_mean","blue_sd"]]  = np.ma.mean(blue), np.ma.std(blue)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return df_colours
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_colours = df_colours