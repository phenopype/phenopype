#%% modules
import copy
import json
import os
from collections import defaultdict
from dataclasses import make_dataclass

import cv2
import numpy as np
import pandas as pd


from phenopype.settings import confirm_options, _annotation_types, flag_verbose
from phenopype.utils_lowlevel import _NoIndent, _NoIndentEncoder, _save_yaml, _load_yaml, _convert_arr_tup_list
from phenopype.utils import save_image


#%% settings   
    
        
#%% functions

def export_csv(annotation, 
               annotation_type=None,
               image_name=None,
               dirpath=None,
               filename="annotations.csv",
               save_suffix=None,
               overwrite=False, 
               **kwargs):

            
    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        dirpath = os.getcwd()
        print('No save directory ("dirpath") specified - using current working directory.')
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return
        
        if save_suffix:
            save_suffix = "_" + save_suffix
        else:
            save_suffix = ""    
    ## annotation copy
    annotation = copy.deepcopy(annotation)
    
    ## filter by annotation type
    if annotation_type.__class__.__name__ == "NoneType":
        print("- no annotation_type selected - exporting all annotations")
        annotation_types = list(_annotation_types.keys())
    elif annotation_type.__class__.__name__ == "str":
        annotation_types = [annotation_type]
    elif annotation_type.__class__.__name__ in [ "list", "CommentedSeq"]:
        annotation_types = annotation_type
        
    ## dry run
    annotation_types1 = []
    for annotation_type in annotation_types:
        if len(annotation[annotation_type].keys()) > 0:
            annotation_types1.append(annotation_type)
    annotation_types = annotation_types1

    for annotation_type in annotation_types:
        
        list_flattened = []
        
        if annotation_type == "comment":
            for annotation_id in annotation[annotation_type].keys():    
                list_flattened.append(
                    pd.DataFrame({
                        **{"image_name": image_name},
                        **{"annotation_type": annotation_type},
                        **{"annotation_id": annotation_id},
                        **{"field": annotation[annotation_type][annotation_id]["data"]["field"]},
                        **{"entry": annotation[annotation_type][annotation_id]["data"]["entry"]},
                        },index=[0])
                    )

        if annotation_type == "contour":
        
            for annotation_id in annotation[annotation_type].keys():    
                idx = 0 
                for coords, support in zip(
                        annotation[annotation_type][annotation_id]["data"]["coord_list"],
                        annotation[annotation_type][annotation_id]["data"]["support"],
                        ):
                    idx += 1
                    list_flattened.append(
                        # df_temp = pd.DataFrame(_convert_arr_tup_list(coords)[0], columns=["x","y"])

                        pd.DataFrame({
                            **{"image_name": image_name},
                            **{"annotation_type": annotation_type},
                            **{"annotation_id": annotation_id},
                            **{"contour_idx": idx},
                            **{"center_x": support["center"][0]},
                            **{"center_y": support["center"][1]}, 
                            **{"area": support["area"]}, 
                            **{"diameter": support["diameter"]}, 
                            **{"hierarchy_level": support["hierarchy_level"]}, 
                            **{"hierarchy_idx_child": support["hierarchy_idx_child"]}, 
                            **{"hierarchy_idx_parent": support["hierarchy_idx_parent"]}
                            }, index=[0])
                        )
        
        if annotation_type == "morphology":
            for annotation_id in annotation[annotation_type].keys():    
                for contour_idx in annotation[annotation_type][annotation_id]["data"]["features"]:
                    list_flattened.append(
                        pd.DataFrame({
                            **{"image_name": image_name},
                            **{"annotation_type": annotation_type},
                            **{"annotation_id": annotation_id},
                            **{"contour_id": annotation[annotation_type][annotation_id]["settings"]["contour_id"]},
                            **{"contour_idx":contour_idx}, 
                            **annotation[annotation_type][annotation_id]["data"]["features"][contour_idx]}, 
                            index=[0])
                        )
                          
        if annotation_type == "texture":
            for annotation_id in annotation[annotation_type].keys():    
                for channel in annotation[annotation_type][annotation_id]["data"]["features"]:
                    for contour_idx in annotation[annotation_type][annotation_id]["data"]["features"][channel]:
                        list_flattened.append(
                            pd.DataFrame({
                                **{"image_name": image_name},
                                **{"annotation_type": annotation_type},
                                **{"annotation_id": annotation_id},
                                **{"channel": channel},
                                **{"contour_id": annotation[annotation_type][annotation_id]["settings"]["contour_id"]},
                                **{"contour_idx":contour_idx}, 
                                **annotation[annotation_type][annotation_id]["data"]["features"][channel][contour_idx]}, 
                                index=[0])
                            )
                          
        if len(list_flattened) > 0:
            print("- " + annotation_type)
            df = pd.concat(list_flattened)    
            filepath = os.path.join(dirpath, annotation_type + save_suffix + ".csv")
            df.to_csv(filepath, index=False)
        else:
            print("- " + annotation_type + ": nothing to save")
    





def load_annotation(filepath, 
                    annotation_type=None,
                    annotation_id=None):
    """

    Parameters
    ----------
    filepath : str
        Path to JSON file containing annotations
    annotation_type : str | list of str, optional
        If file contains multiple annotation types, select one or more to 
        load. None will load all types. The default is None.
    annotation_id : str | list of str, optional
        If file contains multiple annotation IDs, select one or more to 
        load. None will load all IDs within a type. The default is None.

    Returns
    -------
    dict
        Loaded annotations.

    """
    
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
    """  
    
    Parameters
    ----------
    annotation : dict
        Annotation dictionary formatted by phenopype specifications: 
            
        .. code-block:: python

            {
                annotation_type = {
                    annotation_id = annotation,
                    annotation_id = annotation,
                    ...
                    },
                annotation_type = {
                    annotation_id = annotation,
                    annotation_id = annotation,
                    ...
                    },
                ...
            }
            
    annotation_id : str, optional
        String ("a"-"z") specifying the annotation ID to be saved. None will 
        save all IDs. The default is None.
    dirpath : str, optional
        Path to folder where annotation should be saved. None will save the 
        annotation in the current Python working directory. The default is None.
    filename : str, optional
        Filename for JSON file containing annotation. The default is 
        "annotations.json".
    overwrite : bool, optional
        Overwrite options should file or annotation entry in file exist:
            
            - False = Neither file or entry will be overwritten
            - True or "entry" = A single entry will be overwritten
            - "file" = The whole will be overwritten. 
            
        The default is False.

    Returns
    -------
    None.
    
    """
    
    ## kwargs
    indent = kwargs.get("indent", 4)
        
    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        dirpath = os.getcwd()
        print('No save directory ("dirpath") specified - using current working directory.')
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
                print("- overwriting annotation file (overwrite=\"file\")")
        else:
            print("- creating new annotation file")
            pass
        annotation_file = defaultdict(dict)
        break
    
    ## check annotation dict input and convert to type/id/ann structure
    if list(annotation.keys())[0] in _annotation_types.keys():
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
    
    ## remove indents from annotation arrays and lists
    for annotation_type in annotation_file:
        for annotation_id in annotation_file[annotation_type]:    
            for section in annotation_file[annotation_type][annotation_id]:
                for key, value in annotation_file[annotation_type][annotation_id][section].items():
                    
                    ## unindent entries for better legibility
                    if key in ["coord_list", "point_list", "points", "coords"]:
                        if len(value)>0 and not type(value[0]) in [list,tuple, int]:
                            value = [elem.tolist() for elem in value if not type(elem)==list] 
                        value = [_NoIndent(elem) for elem in value]   
                    elif key in ["offset_coords", "channels", "features"]:
                        value = _NoIndent(value)
                    elif key in ["support"]:
                        value_new = []
                        for item in value:
                            item["center"] = _NoIndent(item["center"])
                            value_new.append(item) 
                        value = value_new
                    annotation_file[annotation_type][annotation_id][section][key] = value

    ## save
    with open(filepath, 'w') as file:
        json.dump(annotation_file, 
                    file, 
                    indent=indent, 
                    cls=_NoIndentEncoder)
        
        

def save_ROI(image,
             annotation,
             name,
             dirpath=None,
             prefix=None,
             suffix="roi"): 
    """

    Parameters
    ----------
    image : ndarray
        An image containing regions of interest (ROI).
    annotation : dict
        A phenopype annotation dict containing one or more contour coordinate
        entries.
    name : str
        Name for ROI series (should reflect image content, not "ROI" or the like
        which is specified with prefix or suffix arguments). The contour index
        will be added as a numeric string at the end of the filename.
    dirpath : str, optional
        Path to folder where annotation should be saved. None will save the 
        annotation in the current Python working directory. The default is None.
    prefix : str, optional
        Prefix to prepend to individual ROI filenames. The default is None.
    suffix : str, optional
        Suffix to append to individual ROI filenames. The default is "roi".

    Returns
    -------
    None.

    """
    
    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        dirpath = os.getcwd()
        print('No save directory ("dirpath") specified - using current working directory.')
    else:
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
    """
    

    Parameters
    ----------
    image : ndarray
        A canvas to be saved.
    save_suffix : str
        A suffix to be appended to the filename.
    dirpath : str
        Path to directory to save canvas.

    Returns
    -------
    None.

    """
    
    extension = kwargs.get("extension", ".jpg")
    resize = kwargs.get("resize", 1)
    overwrite = kwargs.get("overwrite", True)
    
    if "." not in extension:
        extension = "." + extension
    name = "canvas_" + save_suffix
    
    save_image(image=image,
               name=name,
               extension=extension,
               dirpath=dirpath,
               resize=resize,
               overwrite=overwrite,
               verbose=flag_verbose)
        
    
        

# def save_reference(annotation, 
#                    overwrite=True, 
#                    dirpath=None, 
#                    active_ref=None):
#     """
    
#     Save a created or detected reference to attributes.yaml
    
#     Parameters
#     ----------
#     annotation: dict
#         A phenopype annotation dict.
#     overwrite: bool optional
#         Overwrite reference if it already exists
#     dirpath: str, optional
#         location to save df
        
#     """
   
#     ## load df
#     px_mm_ratio = annotation["reference"]["a"]["data"]["px_mm_ratio"]


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
#         if not "reference" in attr:
#             attr["reference"] = {}      
#         if not "project_level" in attr["reference"]:
#             attr["reference"]["project_level"] = {}     


#     ## check if file exists
#     if not active_ref.__class__.__name__ == "NoneType":
#         while True:
#             if "reference_detected_px_mm_ratio" in attr["reference"]["project_level"][active_ref] and overwrite == False:
#                 print("- reference not saved (overwrite=False)")
#                 break
#             elif "reference_detected_px_mm_ratio" in attr["reference"]["project_level"][active_ref] and overwrite == True:
#                 print("- save reference to attributes (overwriting)")
#                 pass
#             else:
#                 print("- save reference to attributes")
#                 pass
#             attr["reference"]["project_level"][active_ref]["reference_detected_px_mm_ratio"] = px_mm_ratio
#             break
#         _save_yaml(attr, attr_path)
#     else: 
#         while True:
#             if "reference_manually_measured_px_mm_ratio" in attr["reference"] and overwrite == False:
#                 print("- reference not saved (overwrite=False)")
#                 break
#             elif "reference_manually_measured_px_mm_ratio" in attr["reference"] and overwrite == True:
#                 print("- save reference to attributes (overwriting)")
#                 pass
#             else:
#                 print("- save reference to attributes")
#                 pass
#             attr["reference"]["reference_manually_measured_px_mm_ratio"] = px_mm_ratio
#             break
#         _save_yaml(attr, attr_path)        




