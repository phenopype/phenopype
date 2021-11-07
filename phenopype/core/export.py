#%% modules
import copy
import json
import os
from collections import defaultdict
from dataclasses import make_dataclass

import cv2
import numpy as np
import pandas as pd
import re
from _ctypes import PyObj_FromPtr

from phenopype.settings import confirm_options, _annotation_function_dicts
from phenopype.utils_lowlevel import _save_yaml, _load_yaml, _convert_arr_tup_list
from phenopype.utils import save_image
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
                    if key in ["coord_list", "point_list", "points"]:
                        if type(value) == str:
                            value = eval(value)
                        if annotation_type1 == "contour":
                            value = [np.asarray(elem, dtype=np.int32) for elem in value] 
                        if annotation_type1 == "landmark":
                            value = [tuple(elem) for elem in value] 
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
                print(filepath)
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
            print("- annotation_id missing - please provide an annotation ID [a-z]")
            return
        if not annotation_id.__class__.__name__ == "str":
            annotation_id = str(annotation_id)
        annotation = defaultdict(dict, {annotation["info"]["annotation_type"]:{annotation_id: annotation}})
                        
    ## write annotation to output dict
    for annotation_type in annotation:
        for annotation_id in annotation[annotation_type]:
            annotation_id_new = annotation_id
            if str(annotation_id) in annotation_file[annotation_type]:
                if overwrite in [True, "entry"]:
                    annotation_file[annotation_type][annotation_id_new] = annotation[annotation_type][annotation_id]
                    print("- updating annotation of type \"{}\" with id "
                          "\"{}\" in \"{}\" (overwrite=\"entry\")".format(
                              annotation_type, annotation_id, filename))
                else:
                    print("- annotation of type \"{}\" with id \"{}\" already "
                          "exists in \"{}\" (overwrite=False)".format(
                              annotation_type, annotation_id, filename))
            else:
                annotation_file[annotation_type][annotation_id_new] = annotation[annotation_type][annotation_id]
                print("- writing annotation of type \"{}\" with id "
                      "\"{}\" to \"{}\"".format(
                          annotation_type, annotation_id, filename))  
    
    ## NoIndent annotation arrays and lists
    for annotation_type in annotation_file:
        for annotation_id in annotation_file[annotation_type]:    
            for section in annotation_file[annotation_type][annotation_id]:
                for key, value in annotation_file[annotation_type][annotation_id][section].items():
                    
                    ## unindent entries for better legibility
                    if key in ["coord_list", "point_list", "points", "coords"]:
                        if len(value)>0 and not type(value[0]) in [list,tuple, int]:
                            value = [elem.tolist() for elem in value if not type(elem)==list] 
                        value = [NoIndent(elem) for elem in value]   
                    elif key in ["offset_coords"]:
                        value = NoIndent(value)
                    elif key in ["support"]:
                        value_new = []
                        for item in value:
                            item["center"] = NoIndent(item["center"])
                            value_new.append(item) 
                        value = value_new
                    annotation_file[annotation_type][annotation_id][section][key] = value

    ## save
    with open(filepath, 'w') as file:
        json.dump(annotation_file, 
                    file, 
                    indent=indent, 
                    cls=MyEncoder)
        
        

def save_ROI(image,
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
        
        
def save_canvas(
        image,
        save_suffix,
        dirpath,
        **kwargs):
    
    extension = kwargs.get("extension", ".jpg")
    resize = kwargs.get("resize", 1)
    overwrite = kwargs.get("overwrite", True)
    
    if "." not in extension:
        extension = "." + extension
    name = "canvas_" + save_suffix + extension
    
    save_image(image=image,
               name=name,
               dirpath=dirpath,
               resize=resize,
               overwrite=overwrite)
        
    
        

def save_reference(annotation, 
                   overwrite=True, 
                   dirpath=None, 
                   active_ref=None):
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
   
    ## load df
    px_mm_ratio = annotation["reference"]["a"]["data"]["px_mm_ratio"]


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


    ## check if file exists
    if not active_ref.__class__.__name__ == "NoneType":
        while True:
            if "reference_detected_px_mm_ratio" in attr["reference"]["project_level"][active_ref] and overwrite == False:
                print("- reference not saved (overwrite=False)")
                break
            elif "reference_detected_px_mm_ratio" in attr["reference"]["project_level"][active_ref] and overwrite == True:
                print("- save reference to attributes (overwriting)")
                pass
            else:
                print("- save reference to attributes")
                pass
            attr["reference"]["project_level"][active_ref]["reference_detected_px_mm_ratio"] = px_mm_ratio
            break
        _save_yaml(attr, attr_path)
    else: 
        while True:
            if "reference_manually_measured_px_mm_ratio" in attr["reference"] and overwrite == False:
                print("- reference not saved (overwrite=False)")
                break
            elif "reference_manually_measured_px_mm_ratio" in attr["reference"] and overwrite == True:
                print("- save reference to attributes (overwriting)")
                pass
            else:
                print("- save reference to attributes")
                pass
            attr["reference"]["reference_manually_measured_px_mm_ratio"] = px_mm_ratio
            break
        _save_yaml(attr, attr_path)        



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
