#%% modules
import copy
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import re
from _ctypes import PyObj_FromPtr

from phenopype.settings import confirm_options, _annotation_function_dicts
from phenopype.utils_lowlevel import _save_yaml, _load_yaml, _convert_arr_tup_list

#%% settings   
    
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        if not isinstance(value, (list, tuple, dict)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                    else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        
        if isinstance(obj, np.intc):
            return int(obj)
        
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded
        
#%% functions

def load_annotation(filepath, 
                    annotation_type=None,
                    annotation_id=None):
    
    ## load annotation file
    if os.path.isfile(filepath):
        with open(filepath) as file:
            annotation_file = json.load(file)
        annotation_file = defaultdict(dict, annotation_file)        
    else:
        print("file not found")
        return
        
    ## parse serialized array
    for annotation_type1 in annotation_file:
        for annotation_id1 in annotation_file[annotation_type1]:    
            for section in annotation_file[annotation_type1][annotation_id1]:
                for key, value in annotation_file[annotation_type1][annotation_id1][section].items():
                    if key in ["coord_list"] and type(value) == str:
                        value = eval(value)
                        if annotation_type1 == "contour":
                            value = [np.asarray(elem, dtype=np.int32) for elem in value] 
                    annotation_file[annotation_type1][annotation_id1][section][key] = value
             
    ## subsetting
    while True:
        
        ## filter by annotation type
        if annotation_type.__class__.__name__ == "NoneType":
            print("- no annotation_type selected - returning all annotations")
            annotation = annotation_file
            break
        elif annotation_type.__class__.__name__ == "str":
            annotation_type_list = [annotation_type]
        elif annotation_type.__class__.__name__ == "list":
            annotation_type_list = annotation_type
            pass
        annotation_subset = defaultdict(dict)
        for annotation_type in annotation_type_list:
            annotation_subset[annotation_type] = annotation_file[annotation_type]
                
        ## filter by annotation id
        if annotation_id.__class__.__name__ == "NoneType":
            print("- no annotation_id selected - returning all annotations of type: {}".format(annotation_type_list))
            annotation = annotation_subset
            break
        elif annotation_id.__class__.__name__ == "int":
            annotation_id_list = [str(annotation_id)]
        elif annotation_id.__class__.__name__ == "str":
            annotation_id_list = [annotation_id]
        elif annotation_id.__class__.__name__ == "list":
            annotation_id_list = annotation_id
            pass 
        annotation = defaultdict(dict)
        for annotation_type in annotation_type_list:
            for annotation_id in annotation_id_list:
                annotation_id = int(annotation_id)
                print(annotation_id)
                if annotation_id in annotation_subset[annotation_type]:
                    annotation[annotation_type][annotation_id] = annotation_subset[annotation_type][annotation_id]
        break

    ## return
    return dict(annotation)



def save_annotation(annotation, 
                    annotation_id=None,
                    dirpath=None,
                    filename="annotations.json",
                    overwrite=False, 
                    **kwargs):
    
    ## kwargs
    indent = kwargs.get("indent", 4)
        
    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return
    
    ## filepath
    filepath = os.path.join(dirpath, filename)
    annotation = copy.deepcopy(annotation)
                
    ## open existing json or create new
    while True:
        if os.path.isfile(filepath):
            if overwrite in [False,True,"entry"]:
                with open(filepath) as file:
                    annotation_file = json.load(file)
                annotation_file = defaultdict(dict, annotation_file)           
                print("- loading existing annotation file")
                break
            elif overwrite == "file":
                pass
                print("overwriting annotation file (overwrite=\"file\")")
        else:
            print("- creating new annotation file")
            pass
        annotation_file = defaultdict(dict)
        break
    
    ## check annotation dict input and convert to type/id/ann structure
    if list(annotation.keys())[0] in _annotation_function_dicts.keys():
        annotation = defaultdict(dict, annotation)
    elif list(annotation.keys())[0] == "info":
        if annotation_id.__class__.__name__ == "NoneType":
            print("- annotation_id missing - please provide an integer ID > 0")
            return
        if not annotation_id.__class__.__name__ == "str":
            annotation_id = str(annotation_id)
        annotation = defaultdict(dict, {annotation["info"]["annotation_type"]:{annotation_id: annotation}})
                        
    ## write annotation to output dict
    for annotation_type in annotation:
        for annotation_id in annotation[annotation_type]:
            annotation_id_new = str(annotation_id)
            if str(annotation_id) in annotation_file[annotation_type]:
                if overwrite in [True, "entry"]:
                    annotation_file[annotation_type][annotation_id_new] = annotation[annotation_type][annotation_id]
                    print("- updating annotation of type \"{}\" with id \"{}\" in \"{}\" (overwrite=\"entry\")".format(annotation_type, annotation_id, filename))
                else:
                    print("- annotation of type \"{}\" with id \"{}\" already exists in \"{}\" (overwrite=False)".format(annotation_type, annotation_id, filename))
            else:
                annotation_file[annotation_type][annotation_id_new] = annotation[annotation_type][annotation_id]
                print("- writing annotation of type \"{}\" with id \"{}\" to \"{}\"".format(annotation_type, annotation_id, filename))  
    
    ## NoIndent annotation arrays and lists
    for annotation_type in annotation_file:
        for annotation_id in annotation_file[annotation_type]:    
            for section in annotation_file[annotation_type][annotation_id]:
                for key, value in annotation_file[annotation_type][annotation_id][section].items():
                    if key in ["coord_list"]:
                        if not type(value[0]) == list:
                            value = [elem.tolist() for elem in value if not type(elem)==list] 
                        value = [NoIndent(elem) for elem in value]                             
                        annotation_file[annotation_type][annotation_id][section][key] = value

    ## save
    with open(filepath, 'w') as file:
        json.dump(annotation_file, 
                    file, 
                    indent=indent, 
                    cls=MyEncoder)
        
        

def ROI_save(image,
             annotation,
             dirpath,
             name,
             prefix=None,
             suffix="roi"): 
    
    if not os.path.isdir(dirpath):
        q = input("Save folder {} does not exist - create?.".format(dirpath))
        if q in confirm_options:
            os.makedirs(dirpath)
        else:
            print("Directory not created - aborting")
            return
        
    if prefix is None:
        prefix=""
    else:
        prefix=prefix + "_"
    if suffix is None:
        suffix=""
    else:
        suffix= "_" + suffix
        
    coords = annotation["data"]["coords"]
        
    for idx, roi_coords in enumerate(coords):
        
        rx, ry, rw, rh = cv2.boundingRect(roi_coords)
        roi_rect=image[ry : ry + rh, rx : rx + rw]
                
        roi_name = prefix + name + suffix + "_" + str(idx).zfill(2) + ".tif"
        
        save_path = os.path.join(dirpath, roi_name)
        cv2.imwrite(save_path, roi_rect)
        
        # roi_new_coords = []
        # for coord in roi_coords:
        #     new_coord = [coord[0][0] - rx, coord[0][1] - ry]
        #     roi_new_coords.append([new_coord])
        # roi_new_coords = np.asarray(roi_new_coords, np.int32)
        
        
                    





def save_colours(
    obj_input, overwrite=True, dirpath=None, save_suffix=None, round_digits=1
):
    """
    Save colour intensities to csv. 
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    round_digits: int, optional
        number of digits to round to
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_colours)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## fix na, round, and format to string
    df = df.fillna(-9999)
    df = df.round(round_digits)
    df = df.astype(str)

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "colours_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "colours.csv")

    ## check if file exists
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
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


def save_contours(
    obj_input,
    overwrite=True,
    dirpath=None,
    save_suffix=None,
    save_coords=False,
    convert_coords=True,
    subset=None,
    # round_digits=4
):
    """
    Save contour coordinates and features to csv. This also saves skeletonization
    ouput if the data is contained in the provided DataFrame.
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename
    save_coords: bool, optional
        save the contour coordinates
    convert_coords: bool, optional
        convert the coordinates from array to x and y column
    subset: {"parent", "child"} str, optional
        save only a subset of the parent-child order structure
    round_digits: int, optional
        number of digits to round to
    """

    ## kwargs
    flag_overwrite = overwrite
    flag_convert_coords = convert_coords
    flag_save_coords = save_coords
    flag_subset = subset

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_contours)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No contour df supplied - cannot export contours.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## drop skeleton coordinates
    if "skeleton_coords" in df:
        df.drop(columns="skeleton_coords", inplace=True)

    ## subset
    if flag_subset == "child":
        df = df[df["order"] == "child"]
    elif flag_subset == "parent":
        df = df[df["order"] == "parent"]

    ## convert contour coords to list of tuples
    if flag_convert_coords and flag_save_coords:
        for idx, row in df.iterrows():
            df.at[idx, "coords"] = _convert_arr_tup_list(row["coords"])
        df = df.explode("coords")
        df = pd.concat(
            [
                df.reset_index(drop=True),
                pd.DataFrame(df["coords"].tolist(), columns=["x", "y"]),
            ],
            axis=1,
        )
    else:
        df.drop(columns=["coords"], inplace=True)
        
    ## fix na, round, and format to string
    # df = df.fillna(-9999)
    # df = df.round(round_digits)
    # df = df.astype(str)

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "contours_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "contours.csv")

    ## check if file exists
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
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


# def save_drawing(obj_input, overwrite=True, dirpath=None):
#     """
#     Save drawing coordinates to attributes.yaml 
    
#     Parameters
#     ----------
#     obj_input : DataFrame or container
#         input object
#     overwrite: bool optional
#         overwrite if drawing already exists
#     dirpath: str, optional
#         location to save df

#     """
#     ## kwargs
#     flag_overwrite = overwrite

#     ## load df
#     if obj_input.__class__.__name__ == "DataFrame":
#         df = obj_input
#     elif obj_input.__class__.__name__ == "container":
#         df = obj_input.df_draw
#         if (
#             dirpath.__class__.__name__ == "NoneType"
#             and not obj_input.dirpath.__class__.__name__ == "NoneType"
#         ):
#             dirpath = obj_input.dirpath
#     else:
#         print("No df supplied - cannot export results.")
#         return

#     ## dirpath
#     if dirpath.__class__.__name__ == "NoneType":
#         print('No save directory ("dirpath") specified - cannot save result.')
#         return
#     else:
#         if not os.path.isdir(dirpath):
#             q = input("Save folder {} does not exist - create?.".format(dirpath))
#             if q in ["True", "true", "y", "yes"]:
#                 os.makedirs(dirpath)
#             else:
#                 print("Directory not created - aborting")
#                 return

#     ## load attributes file
#     attr_path = os.path.join(dirpath, "attributes.yaml")
#     if os.path.isfile(attr_path):
#         attr = _load_yaml(attr_path)
#     else:
#         attr = {}
#     if not "drawing" in attr:
#         attr["drawing"] = {}

#     ## check if file exists
#     while True:
#         if "drawing" in attr and flag_overwrite == False:
#             print("- drawing not saved (overwrite=False)")
#             break
#         elif "drawing" in attr and flag_overwrite == True:
#             print("- drawing saved (overwriting)")
#             pass
#         elif not "drawing" in attr:
#             attr["drawing"] = {}
#             print("- drawing saved")
#             pass
#         for idx, row in df.iterrows():
#             # if not row["coords"] == attr["drawing"]["coords"]:
#             attr["drawing"] = dict(row)
#         _save_yaml(attr, attr_path)
#         break
    
def save_drawings(obj_input, overwrite=True, dirpath=None, save_suffix=None):
    """
    Save drawing coordinates and information ("include"" and "label") to csv.

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename

    """
    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "ndarray":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_drawings
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No drawing df supplied - cannot save drawing.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "drawings_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "drawings.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- drawings not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- drawings saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- drawings saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break



def save_data_entry(obj_input, overwrite=True, dirpath=None):
    """
    Save data entry to attributes.yaml 

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite if entry already exists
    dirpath: str, optional
        location to save df
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_other_data
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
    else:
        print("No df supplied - cannot export results.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## load attributes file
    attr_path = os.path.join(dirpath, "attributes.yaml")
    if os.path.isfile(attr_path):
        attr = _load_yaml(attr_path)
    else:
        attr = {}
    if not "other" in attr:
        attr["other"] = {}

    ## check if entry exists
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


def save_landmarks(obj_input, overwrite=True, dirpath=None, save_suffix=None):
    """
    Save landmark coordinates to csv.

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_landmarks
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if not save_suffix.__class__.__name__ == "NoneType":
        path = os.path.join(dirpath, "landmarks_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "landmarks.csv")

    ## check if file exists
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
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


def save_masks(obj_input, overwrite=True, dirpath=None, save_suffix=None):
    """
    Save mask coordinates and information ("include"" and "label") to csv.

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename

    """
    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_masks
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No mask df supplied - cannot save mask.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "masks_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "masks.csv")

    ## check if file exists
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
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


def save_polylines(obj_input, overwrite=True, dirpath=None, save_suffix=None):
    """
    Save polylines to csv.

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_polylines
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "polylines_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "polylines.csv")

    ## check if file exists
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
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


def save_reference(obj_input, overwrite=True, dirpath=None, active_ref=None):
    """
    Save a created or detected reference to attributes.yaml
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    """

    ## kwargs
    flag_overwrite = overwrite
    
    ## load df
    if obj_input.__class__.__name__ in ["int", "float"]:
        px_mm_ratio = obj_input
    elif obj_input.__class__.__name__ == "container":
        if hasattr(obj_input, "reference_detected_px_mm_ratio"):
            px_mm_ratio = obj_input.reference_detected_px_mm_ratio
        if hasattr(obj_input, "reference_manually_measured_px_mm_ratio") and obj_input.reference_manual_mode==True:
            px_mm_ratio = obj_input.reference_manually_measured_px_mm_ratio
            active_ref = None
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
    else:
        print("No reference supplied - cannot export results.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## load attributes file        
    attr_path = os.path.join(dirpath, "attributes.yaml")
    if os.path.isfile(attr_path):
        attr = _load_yaml(attr_path)
        if not "reference" in attr:
            attr["reference"] = {}      
        if not "project_level" in attr["reference"]:
            attr["reference"]["project_level"] = {}     

    print(active_ref)
    ## check if file exists
    if not active_ref.__class__.__name__ == "NoneType":
        while True:
            if "detected_px_mm_ratio" in attr["reference"]["project_level"][active_ref] and flag_overwrite == False:
                print("- reference not saved (overwrite=False)")
                break
            elif "detected_px_mm_ratio" in attr["reference"]["project_level"][active_ref] and flag_overwrite == True:
                print("- save reference to attributes (overwriting)")
                pass
            else:
                print("- save reference to attributes")
                pass
            attr["reference"]["project_level"][active_ref]["detected_px_mm_ratio"] = px_mm_ratio
            break
        _save_yaml(attr, attr_path)
    else: 
        while True:
            if "manually_measured_px_mm_ratio" in attr["reference"] and flag_overwrite == False:
                print("- reference not saved (overwrite=False)")
                break
            elif "manually_measured_px_mm_ratio" in attr["reference"] and flag_overwrite == True:
                print("- save reference to attributes (overwriting)")
                pass
            else:
                print("- save reference to attributes")
                pass
            attr["reference"]["manually_measured_px_mm_ratio"] = px_mm_ratio
            break
        _save_yaml(attr, attr_path)        


def save_textures(
    obj_input, overwrite=True, dirpath=None, save_suffix=None, round_digits=1,
    name=""
):
    """
    Save colour intensities to csv. 
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    round_digits: int, optional
        number of digits to round to
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_textures)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## fix na, round, and format to string
    # df = df.fillna(-9999)
    df = df.round(round_digits)
    df = df.astype(str)

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if len(name) > 0:
        name = "_" + name
    if save_suffix:
        path = os.path.join(dirpath, "textures" + name + "_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "textures" + name + ".csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- textures not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- textures saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- textures saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False, na_rep='NA')
        break

def save_shapes(
    obj_input, overwrite=True, dirpath=None, save_suffix=None, round_digits=1
):
    """
    Save colour intensities to csv. 
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    round_digits: int, optional
        number of digits to round to
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_shapes)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## fix na, round, and format to string
    # df = df.fillna(-9999)
    df = df.round(round_digits)
    df = df.astype(str)

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "shapes_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "shapes.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- shapes not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- shapes saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- shapes saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False, na_rep='NA')
        break