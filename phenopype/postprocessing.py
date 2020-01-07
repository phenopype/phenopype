#%% modules
import cv2
import copy
import numpy as np
import os
import sys 

from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours
from phenopype.utils import show_img, load_yaml, save_yaml, show_yaml
from phenopype.utils_lowlevel import _image_viewer, _load_image

#%% methods

def save_csv(obj_input, **kwargs):
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
    keep = kwargs.get("keep",None)
    ## load image
    image, flag_input = _load_image(obj_input)

    if flag_input == "pype_container":
        dirpath = obj_input.dirpath
        df = obj_input.df_result
        name = df["pype_name"].iloc[0]
# print (df[(df == 'something1').all(1)])

    if keep:
        df = df.loc[df["order"]=="parent"]

    df = df.fillna(-9999)
    df = df.round(1)
    df = df.astype(str)

    save_path = os.path.join(dirpath, name + "_result.csv")

    if os.path.exists(save_path):
        if flag_overwrite == True:
            df.to_csv(path_or_buf=save_path, sep=",")
    else:
        df.to_csv(path_or_buf=save_path, sep=",")
        
        
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
    """
    ## kwargs
    flag_overwrite = kwargs.get("overwrite", False)
    resize = kwargs.get("resize", 1)
        
    ## load image
    image, flag_input = _load_image(obj_input)

    if flag_input == "pype_container":
        dirpath = obj_input.dirpath
        img = obj_input.canvas
        name = obj_input.df_result["pype_name"].iloc[0] + "_result"

    img = cv2.resize(img, (0,0), fx=1*resize, fy=1*resize) 


    save_path = os.path.join(dirpath, name + ".jpg")

    if os.path.exists(save_path):
        if flag_overwrite == True:
            cv2.imwrite(save_path, img)
    else:
        cv2.imwrite(save_path, img)
