    #%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours
from phenopype.utils import * #load_image, load_meta_data, show_image, save_image
from phenopype.utils_lowlevel import _image_viewer, _save_yaml, _load_yaml, _contours_arr_tup

#%% functions

def save_canvas(obj_input, **kwargs):
    """Save a pandas dataframe to csv. 
    
    Parameters
    ----------
    df: df
        object_finder outpur (pandas data frame) to save
    name: str
        name for saved df
    save_dir: str
        location to save df
    append: str (optional)
        append df name with string to prevent overwriting
    overwrite: bool (optional, default: False)
        overwrite df if name exists
    silent: bool (optional, default: True)
        do not print where file was saved
    """
    ## kwargs
    flag_overwrite = kwargs.get("overwrite", True)
    dirpath = kwargs.get("directory", None)
    save_suffix = kwargs.get("save_suffix", None)
    resize = kwargs.get("resize", 0.5)

    ## load df
    if obj_input.__class__.__name__ == 'ndarray':
        image = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot save canvas.")
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        save_suffix = obj_input.save_suffix
        dirpath = obj_input.dirpath
    else:
        warnings.warn("No image supplied - cannot save canvas.")

    ## resize
    if resize < 1:
        image = cv2.resize(image, (0,0), fx=1*resize, fy=1*resize) 

    ## save
    if save_suffix:
        path = os.path.join(dirpath, "canvas_" + save_suffix + ".jpg")
    else:
        path = os.path.join(dirpath, "canvas.jpg")
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- canvas not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- canvas saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- canvas saved under " + path + ".")
            pass
        cv2.imwrite(path, image)
        break




def save_colours(obj_input, **kwargs):
    """Save a pandas dataframe to csv. 
    
    Parameters
    ----------
    name: str (optional, default: "results")
        name for saved csv
    dirpath: str (default: None)
        location to save df
    round: int (optional, default: 1)
        number of digits to round to
    overwrite: bool (optional, default: False)
        overwrite csv if it already exists
    silent: bool (optional, default: True)
        do not print where file was saved
    """
    ## kwargs
    flag_overwrite = kwargs.get("overwrite", True)
    dirpath = kwargs.get("directory", None)
    round_digits = kwargs.get("round",1)
    save_suffix = kwargs.get("save_suffix", None)

    ## load df
    if obj_input.__class__.__name__ == 'DataFrame':
        df = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot export results.")
    elif obj_input.__class__.__name__ == "container":
        if not dirpath:
            dirpath = obj_input.dirpath
        df = obj_input.df_colours
        save_suffix = obj_input.save_suffix
    else:
        warnings.warn("No df supplied - cannot export results.")

    ## fix na, round, and format to string
    df = df.fillna(-9999)
    df = df.round(round_digits)
    df = df.astype(str)

    ## save
    if save_suffix:
        path = os.path.join(dirpath, "colours_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "colours.csv")
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- colours not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- colours saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- colours saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",",index=False)
        break


def save_contours(obj_input, **kwargs):
    """Save a pandas dataframe to csv. 
    
    Parameters
    ----------
    df: df
        object_finder outpur (pandas data frame) to save
    name: str
        name for saved df
    save_dir: str
        location to save df
    append: str (optional)
        append df name with string to prevent overwriting
    overwrite: bool (optional, default: False)
        overwrite df if name exists
    """
    ## kwargs
    flag_overwrite = kwargs.get("overwrite", True)
    dirpath = kwargs.get("dirpath", None)
    save_suffix = kwargs.get("save_suffix", None)
    convert_coords = kwargs.get("convert_coords", True)

    ## load df
    if obj_input.__class__.__name__ == 'DataFrame':
        df = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot save contours.")
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_contours)
        dirpath = obj_input.dirpath
        save_suffix = obj_input.save_suffix
    else:
        warnings.warn("No df supplied - cannot export contours.")

    ## convert contour coords to list of tuples
    if convert_coords:
        for idx, row in df.iterrows():
            df.at[idx,"coords"] = _contours_arr_tup(row["coords"])
        df = df.explode("coords")
        df[["x","y"]] = pd.DataFrame(df["coords"].tolist(), columns=["x","y"])
        df.drop(columns="coords", inplace=True)

    ## save
    if save_suffix:
        path = os.path.join(dirpath, "contours_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "contours.csv")
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- contours not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- contours saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- contours saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",",index=False)
        break



def save_data_entry(obj_input, **kwargs):
    """Save a pandas dataframe to csv. 
    
    Parameters
    ----------
    name: str (optional, default: "results")
        name for saved csv
    dirpath: str (default: None)
        location to save df
    round: int (optional, default: 1)
        number of digits to round to
    overwrite: bool (optional, default: False)
        overwrite csv if it already exists
    silent: bool (optional, default: True)
        do not print where file was saved
    """
    ## kwargs
    flag_overwrite = kwargs.get("overwrite", True)
    dirpath = kwargs.get("directory", None)
    round_digits = kwargs.get("round",1)
    save_suffix = kwargs.get("save_suffix", None)

    ## load df
    if obj_input.__class__.__name__ == 'DataFrame':
        df = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot export results.")
    elif obj_input.__class__.__name__ == "container":
        if not dirpath:
            dirpath = obj_input.dirpath
        df = obj_input.df_other_data
    else:
        warnings.warn("No df supplied - cannot export results.")

    attr_path = os.path.join(dirpath, "attributes.yaml")
    attr = _load_yaml(attr_path)
    if not "other" in attr:
        attr["other"] = {}

    while True:
        for col in list(df):
            if col in attr["other"] and flag_overwrite == False:
                print("- column " + col + " not saved (overwrite=False)")
                continue
            elif col in attr["other"] and flag_overwrite == True:
                print("- add column " + col + " (overwriting)")
                pass
            else:
                print("- add column " + col)
                pass
            attr["other"][col] = df[col][0]
        break

    _save_yaml(attr, attr_path)


def save_landmarks(obj_input, **kwargs):
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
    flag_overwrite = kwargs.get("overwrite", True)
    dirpath = kwargs.get("directory", None)
    save_suffix = kwargs.get("save_suffix", None)

    ## load df
    if obj_input.__class__.__name__ == 'DataFrame':
        df = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot save landmarks.")
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_landmarks
        dirpath = obj_input.dirpath
        save_suffix = obj_input.save_suffix
    else:
        warnings.warn("No df supplied - cannot export results.")

    ## save
    if save_suffix:
        path = os.path.join(dirpath, "landmarks_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "landmarks.csv")
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- landmarks not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- landmarks saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- landmarks saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",",index=False)
        break



def save_masks(obj_input, **kwargs):
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
    flag_overwrite = kwargs.get("overwrite", True)
    dirpath = kwargs.get("directory", None)
    save_suffix = kwargs.get("save_suffix", None)
    
    ## load df
    if obj_input.__class__.__name__ == 'ndarray':
        df = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot save masks.")
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_masks
        save_suffix = obj_input.save_suffix
        dirpath = obj_input.dirpath
    else:
        warnings.warn("No mask df supplied - cannot save mask.")
    
    ## save
    if save_suffix:
        path = os.path.join(dirpath, "masks_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "masks.csv")
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- masks not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- masks saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- masks saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",",index=False)
        break



def save_polylines(obj_input, **kwargs):
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
    flag_overwrite = kwargs.get("overwrite", True)
    dirpath = kwargs.get("directory", None)
    save_suffix = kwargs.get("save_suffix", None)

    ## load df
    if obj_input.__class__.__name__ == 'DataFrame':
        df = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot save polylines.")
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_polylines
        dirpath = obj_input.dirpath
        save_suffix = obj_input.save_suffix
    else:
        warnings.warn("No df supplied - cannot export results.")

    ## save
    if save_suffix:
        path = os.path.join(dirpath, "polylines_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "polylines.csv")
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- polylines not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- polylines saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- polylines saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",",index=False)
        break
