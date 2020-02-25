#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours
from phenopype.utils import load_image, load_meta_data, show_image, save_image
from phenopype.utils_lowlevel import _image_viewer, _save_yaml

#%% functions

def save_results(obj_input, **kwargs):
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
    flag_overwrite = kwargs.get("overwrite", False)
    dirpath = kwargs.get("directory", None)
    df = kwargs.get("df", None)
    name = kwargs.get("name","results")
    round_digits = kwargs.get("round",1)
    silent = kwargs.get("silent", True)
    
    ## load df
    if obj_input.__class__.__name__ == 'DataFrame':
        df = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot export results.")
    elif obj_input.__class__.__name__ == "container":
        if not dirpath:
            dirpath = obj_input.dirpath
        df = obj_input.df
    else:
        warnings.warn("No df supplied - cannot export results.")

    ## fix na, round, and format to string
    df = df.fillna(-9999)
    df = df.round(round_digits)
    df = df.astype(str)

    ## save
    save_path = os.path.join(dirpath,  name + ".csv")
    if os.path.exists(save_path):
        if flag_overwrite == True:
            df.to_csv(path_or_buf=save_path, sep=",",index=False)
            if not silent:
                print("Results saved under " + save_path + " (overwritten).")
        else:
            if not silent:
                print("Results not saved - file already exists (overwrite = False).")
    else:
        df.to_csv(path_or_buf=save_path, sep=",",index=False)
        if not silent:
            print("Results saved under " + save_path + ".")



def save_overlay(obj_input, **kwargs):
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
    dirpath = kwargs.get("directory", None)
    flag_overwrite = kwargs.get("overwrite", False)
    name = kwargs.get("name","results")
    resize = kwargs.get("resize", 1)
    silent = kwargs.get("silent", True)
    
    ## load df
    if obj_input.__class__.__name__ == 'ndarray':
        image = obj_input
        if not dirpath:
            warnings.warn("No save directory specified - cannot save overlay.")
    elif obj_input.__class__.__name__ == "container":
        if not dirpath:
            dirpath = obj_input.dirpath
        image = obj_input.canvas
    else:
        warnings.warn("No image supplied - cannot save overlay.")

    ## resize
    if resize < 1:
        image = cv2.resize(image, (0,0), fx=1*resize, fy=1*resize) 

    ## save
    save_path = os.path.join(dirpath,  name + ".jpg")
    if os.path.exists(save_path):
        if flag_overwrite == True:
            cv2.imwrite(save_path, image)
            if not silent:
                print("Results saved under " + save_path + " (overwritten).")
        else:
            if not silent:
                print("Results not saved - file already exists (overwrite = False).")
    else:
        cv2.imwrite(save_path, image)
        if not silent:
            print("Results saved under " + save_path + ".")

        
        
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
    flag_overwrite = kwargs.get("overwrite", False)
    dirpath = kwargs.get("dirpath", None)
    df = kwargs.get("df", None)
        
    ## load df
    if obj_input.__class__.__name__ == "ndarray":
        if not dirpath:
            warnings.warn("No save directory specified - cannot export results.")
        elif not df:
            warnings.warn("No df supplied - cannot export results.")
    elif obj_input.__class__.__name__ == "container":
        if not dirpath:
            dirpath = obj_input.dirpath
        df = obj_input.df

    obj_output = {}
    obj_output["image"] = obj_input.image_data
    obj_output["contours"] = {}
    
    for contour in obj_input.contours.keys():
        contour_dict = {}
        contour_dict["label"] = contour
        contour_dict["center"] = str((obj_input.contours[contour]["x"], obj_input.contours[contour]["y"]))
        contour_dict["order"] = str(obj_input.contours[contour]["order"])
        contour_dict["idx_child"] = 1 # str(obj_input.contours[contour]["idx_child"])
        contour_dict["idx_parent"] = str(obj_input.contours[contour]["idx_parent"])
        x_coords, y_coords = [], []
        for coord in obj_input.contours[contour]["coords"]:
            x_coords.append(coord[0][0])
            y_coords.append(coord[0][1])
        contour_dict["x_coords"], contour_dict["y_coords"] = str(x_coords), str(y_coords)
        obj_output["contours"][contour] = contour_dict

    save_path = os.path.join(dirpath, "contours.yaml")

    if os.path.exists(save_path):
        if flag_overwrite == True:
             _save_yaml(obj_output, save_path)
    else:
         _save_yaml(obj_output, save_path)
