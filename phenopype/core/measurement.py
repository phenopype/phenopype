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

    ## timestamp
    timestamp = datetime.today().strftime('%Y:%m:%d %H:%M:%S')

    ## load image
    df_landmarks = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"})
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_landmarks"):
            df_landmarks = obj_input.df_landmarks
    else:
        warnings.warn("wrong input format.")
        return

    ## more kwargs
    point_size = kwargs.get("point_size", _auto_point_size(image))
    point_col = kwargs.get("point_col", "red")
    label_size = kwargs.get("label_size", _auto_text_size(image))
    label_width = kwargs.get("label_width", _auto_text_width(image))
    label_col = kwargs.get("label_col", "black")

    ## method
    while True:
        if not df_landmarks.__class__.__name__ == "NoneType" and flag_overwrite == False:
            warnings.warn("landmarks already set (overwrite=False)")
            for label, x, y in zip(df_landmarks.landmark, df_landmarks.x, df_landmarks.y):
                cv2.circle(image, (x,y), point_size, colours[point_col], -1)
                cv2.putText(image, str(label), (x,y), 
                    cv2.FONT_HERSHEY_SIMPLEX, label_size, colours[label_col], label_width, cv2.LINE_AA)
            break
        elif not df_landmarks.__class__.__name__ == "NoneType" and flag_overwrite == True:
            print("- set landmarks (overwrite)")
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
        image = out.image_copy

        ## make df
        df_landmarks = pd.DataFrame(coords, columns=['x', 'y'])
        df_landmarks.reset_index(inplace=True)
        df_landmarks.rename(columns={"index": "landmark"},inplace=True)
        df_landmarks["landmark"] = df_landmarks["landmark"] + 1
        df_image_data["date_phenopyped"] = timestamp
        df_landmarks = pd.concat([pd.concat([df_image_data]*len(df_landmarks)).reset_index(drop=True), 
                                df_landmarks.reset_index(drop=True)], axis=1)
        break

    ## return
    if obj_input.__class__.__name__ == "ndarray":
            return image, df_landmarks
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_landmarks = df_landmarks
        obj_input.ov_landmarks = flag_overwrite
        obj_input.image = image



def polyline(obj_input, **kwargs):
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

    ## timestamp
    timestamp = datetime.today().strftime('%Y:%m:%d %H:%M:%S')

    ## load image
    df_polylines = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"})
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_polylines"):
            df_polylines = obj_input.df_polylines
    else:
        warnings.warn("wrong input format.")
        return

    ## more kwargs
    line_width = kwargs.get("line_width", _auto_line_width(image))

    ## method
    while True:
        if not df_polylines.__class__.__name__ == "NoneType" and flag_overwrite == False:
            warnings.warn("polylines already drawn (overwrite=False)")
            for x_coords, y_coords in zip(df_polylines.x_coords, df_polylines.y_coords):
                point_list = list(zip(eval(x_coords), eval(y_coords)))
                cv2.polylines(image, np.array([point_list]), 
                              False, colours["green"], line_width)
            break
        elif not df_polylines.__class__.__name__ == "NoneType" and flag_overwrite == True:
            print("- set polylines (overwrite)")
            pass
        elif df_polylines.__class__.__name__ == "NoneType":
            print("- set polylines")
            pass
        
        out = _image_viewer(image, tool="polyline")
        image = out.image_copy
        idx = 0
        df_polylines = pd.DataFrame(columns=["polyline", "length", "x_coords", "y_coords"])
        if len(out.point_list) > 0:
            for point_list in out.point_list:
                idx += 1
                x_coords, y_coords = [], []
                arc = np.array(point_list)
                arc_length = int(cv2.arcLength(arc, closed=False))
                for point in point_list:
                    x_coords.append(point[0])
                    y_coords.append(point[1])
                poly_line = {"polyline": idx,
                             "length": arc_length,
                             "x_coords": str(x_coords),
                             "y_coords": str(y_coords)}
                df_polylines = df_polylines.append(poly_line, ignore_index=True, sort=False)
        df_image_data["date_phenopyped"] = timestamp
        df_polylines = pd.concat([pd.concat([df_image_data]*len(df_polylines)).reset_index(drop=True), 
                                df_polylines.reset_index(drop=True)], axis=1)
        break 

    ## return
    if obj_input.__class__.__name__ == "ndarray":
            return image, df_polylines
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_polylines = df_polylines
        obj_input.ov_polylines = flag_overwrite
        obj_input.image = image

def colour(obj_input, **kwargs):

    ## kwargs
    channels = kwargs.get("channels", ["gray"])
    contour_dict = kwargs.get("contours", None)
    contour_df = kwargs.get("df", None)
    
    ## load image and contours
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if not contour_dict:
            sys.exit("no contours provided")
        if contour_df.__class__.__name__ == "NoneType":
            warnings.warn("no data-frame for contours provided")
            contour_df = pd.DataFrame({"filename":"unknown"}, index=[0]).T
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image_copy
        contour_dict = obj_input.contours
        contour_df = obj_input.df

    ## create forgeround mask
    image_bin = np.zeros(image.shape[:2], np.uint8)
    for label, contour in contour_dict.items():
        if contour["order"]=="parent":
            image_bin = cv2.fillPoly(image_bin, [contour["coords"]], 255)
        elif contour["order"]=="child":
            image_bin = cv2.fillPoly(image_bin, [contour["coords"]], 0)

    foreground_mask = np.invert(np.array(image_bin, dtype=np.bool))

    ## method
    if "gray" in channels:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_cols = {"gray_mean":"NA",
                    "gray_sd":"NA"}
        contour_df = contour_df.assign(**new_cols)
        for label, contour in contour_dict.items():
            rx,ry,rw,rh = cv2.boundingRect(contour["coords"])
            grayscale =  ma.array(data=image_gray[ry:ry+rh,rx:rx+rw], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            contour_df.loc[contour_df["contour"]==label,["gray_mean","gray_sd"]] = np.ma.mean(grayscale), np.ma.std(grayscale)

    if "rgb" in channels:
        contour_df = contour_df.assign(**{"red_mean":"NA",
                           "red_sd":"NA",
                           "green_mean":"NA",
                           "green_sd":"NA",
                           "blue_mean":"NA",
                           "blue_sd":"NA"})
        for label, contour in contour_dict.items():
            rx,ry,rw,rh = cv2.boundingRect(contour["coords"])
            blue =  ma.array(data=image[ry:ry+rh,rx:rx+rw,0], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            green =  ma.array(data=image[ry:ry+rh,rx:rx+rw,1], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            red =  ma.array(data=image[ry:ry+rh,rx:rx+rw,2], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            contour_df.loc[contour_df["contour"]==label,["red_mean","red_sd"]]  = np.ma.mean(red), np.ma.std(red)
            contour_df.loc[contour_df["contour"]==label,["green_mean","green_sd"]]  = np.ma.mean(green), np.ma.std(green)
            contour_df.loc[contour_df["contour"]==label,["blue_mean","blue_sd"]]  = np.ma.mean(blue), np.ma.std(blue)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return contour_df
    elif obj_input.__class__.__name__ == "container":
        obj_input.df = contour_df