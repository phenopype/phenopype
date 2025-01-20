#%% modules

from collections import defaultdict
import copy
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

from phenopype import __version__
from phenopype import _vars
from phenopype import config
from phenopype import decorators
from phenopype import utils
from phenopype import utils_lowlevel as ul
from phenopype.core import preprocessing

#%% functions

def convert_annotation(
    annotations,
    annotation_type,
    annotation_id,
    annotation_type_new,
    annotation_id_new,
    overwrite=False,
    **kwargs
):
    """
    convert coordinates from one annotation type to another. currently, only
    converting from contour to mask format is possible

    Parameters
    ----------

    annotations : dict
        A phenopype annotation dictionary.
    annotation_type : str | list of str
        If dict contains multiple annotation types, select one or more to
        load. None will load all types.
    annotation_id : str | list of str
        If file contains multiple annotation IDs, select one or more to
        load. None will load all IDs within a type.
    annotation_type_new : str
       target annotation type
    annotation_id_new : str | list of str, optional
       target annotation id
    overwrite : bool, optional
        if target exists, overwrite? The default is False.

    Returns
    -------
    None.


    """

    # =============================================================================
    # annotation management

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
        prep_msg="- extracting annotation:",
    )

    # =============================================================================
    # setup

    ## convert_annotation
    fun_name = sys._getframe().f_code.co_name

    # =============================================================================
    # method

    if annotation_type_new == _vars._mask_type:

        ## method setup
        if "label" in annotation["data"]:
            label = annotation["data"]["label"]
        else:
            label = kwargs.get("label", "mask1")
        include = kwargs.get("include", True)

        annotation_old_data = annotation["data"][annotation_type]

        annotation_new_data = []
        for idx, coord_list in enumerate(annotation_old_data):
            annotation_new_data.append(ul._convert_arr_tup_list(coord_list,  add_first=True))


        annotation = {
            "info": {
                "annotation_type": annotation_type_new,
                "phenopype_function": fun_name,
                "phenopype_version": __version__,
            },
            "settings": {
                "tool": "polygon",
            },
            "data": {
                "label": label,
                "include": include,
                "n": len(annotation_new_data),
                annotation_type_new: annotation_new_data,
            },
        }

    return ul._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type_new,
        annotation_id=annotation_id_new,
        kwargs=kwargs,
    )




def export_csv(
    annotations,
    dir_path,
    annotation_type=None,
    image_name=None,
    overwrite=False,
    **kwargs
):
    """
    export annotations from json to csv format.

    Parameters
    ----------
    annotations : dict
        A phenopype annotation dictionary.
    dir_path : str
        To which folder should the csv file be exported.
        PYPE: Automatically set to the current image directory
    annotation_type : str / list, optional
        Which annotation types should be exported - can be string or list of strings.
    image_name : str
       Image name to be added as a column
       PYPE: Automatically adds the image name.
    overwrite : bool, optional
        Should an existing csv file be overwritten. The default is False.


    Returns
    -------
    None.

    """

    ## dirpath
    if not image_name:
        ul._print(
            "Warning: missing image_name argument - exported CSV will not contain information about source-image",
            lvl=1
        )

    ## file name formatting
    if kwargs.get("save_prefix"):
        save_prefix = kwargs.get("save_prefix") + "_"
    else:
        save_prefix = ""

    if kwargs.get("save_suffix"):
        save_suffix = "_" + kwargs.get("save_suffix")
    else:
        save_suffix = ""

    ## annotations copy
    annotations = copy.deepcopy(annotations)

    ## format annotation type
    if annotation_type.__class__.__name__ == "NoneType":
        ul._print("- no annotation_type selected - exporting all annotations")
        annotation_types = list(annotations.keys())
    elif annotation_type.__class__.__name__ == "str":
        annotation_types = [annotation_type]
    elif annotation_type.__class__.__name__ in ["list", "CommentedSeq"]:
        annotation_types = annotation_type

    # ## dry run
    # annotation_types1 = []
    # for annotation_type in annotation_types:
    #     if len(annotations[annotation_type].keys()) > 0:
    #         annotation_types1.append(annotation_type)
    # annotation_types = annotation_types1

    for annotation_type in annotation_types:

        list_flattened = []

        ## comment
        if annotation_type == _vars._comment_type and annotation_type in annotations.keys():
            for annotation_id in annotations[annotation_type].keys():

                label = annotations[annotation_type][annotation_id]["data"]["label"]
                comment = annotations[annotation_type][annotation_id]["data"][annotation_type]
                list_flattened.append(
                    pd.DataFrame(
                        {
                            **{"image_name": image_name},
                            **{"annotation_type": annotation_type},
                            **{"annotation_id": annotation_id},
                            **{label:comment},
                        },
                        index=[0],
                    )
                )

        ## contour
        if annotation_type == _vars._contour_type:
            for annotation_id in annotations[annotation_type].keys():
                for idx, (coords, support) in enumerate(zip(
                    annotations[annotation_type][annotation_id]["data"][annotation_type],
                    annotations[annotation_type][annotation_id]["data"]["support"],
                ), 1):
                    list_flattened.append(
                        # df_temp = pd.DataFrame(ul._convert_arr_tup_list(coords)[0], columns=["x","y"])
                        pd.DataFrame(
                            {
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
                                **{"hierarchy_idx_parent": support["hierarchy_idx_parent"]},
                            },
                            index=[0],
                        )
                    )


        ## landmark
        if annotation_type == _vars._landmark_type:
            for annotation_id in annotations[annotation_type].keys():
                lm_tuple_list = list(zip(*annotations[annotation_type][annotation_id]["data"][annotation_type]))
                list_flattened.append(
                    pd.DataFrame.from_dict(
                        {
                            **{"image_name": image_name},
                            **{"annotation_type": annotation_type},
                            **{"annotation_id": annotation_id},
                            **{"landmark_idx": range(1,len(lm_tuple_list[0])+1)},
                            **{"x_coords": lm_tuple_list[0]},
                            **{"y_coords": lm_tuple_list[1]},

                        },
                    )
                )

        ## polyline
        if annotation_type == _vars._line_type:
            for annotation_id in annotations[annotation_type].keys():
                for idx, (coords, length) in enumerate(zip(
                    annotations[annotation_type][annotation_id]["data"][annotation_type],
                    annotations[annotation_type][annotation_id]["data"]["lengths"],
                ), 1):
                    # line_tuple_list = list(zip(*coords))
                    list_flattened.append(
                        pd.DataFrame(
                            {
                                **{"image_name": image_name},
                                **{"annotation_type": annotation_type},
                                **{"annotation_id": annotation_id},
                                **{"line_idx": idx},
                                # **{"node_idx": range(1,len(line_tuple_list[0])+1)},
                                # **{"x_coords": line_tuple_list[0]},
                                # **{"y_coords": line_tuple_list[1]},
                                **{"length": length},
                            },
                            index=[0],
                        )
                    )

        ## reference
        if annotation_type == _vars._reference_type:
            for annotation_id in annotations[annotation_type].keys():
                distance = annotations[annotation_type][annotation_id]["data"][annotation_type][0]
                unit = annotations[annotation_type][annotation_id]["data"][annotation_type][1]
                list_flattened.append(
                    pd.DataFrame(
                        {
                            **{"image_name": image_name},
                            **{"annotation_type": annotation_type},
                            **{"annotation_id": annotation_id},
                            **{"distance": distance},
                            **{"unit": unit},
                        },
                        index=[0],
                    )
                )


        ## shape_features
        if annotation_type == _vars._shape_feature_type:
            for annotation_id in annotations[annotation_type].keys():
                contour_id = annotations[annotation_type][annotation_id]["settings"][_vars._contour_type + "_id"]
                for idx, annotation in enumerate(
                    annotations[annotation_type][annotation_id]["data"][annotation_type], 1
                ):
                    list_flattened.append(
                        pd.DataFrame(
                            {
                                **{"image_name": image_name},
                                **{"annotation_type": annotation_type},
                                **{"annotation_id": annotation_id},
                                **{"contour_id": contour_id},
                                **{"contour_idx": idx},
                                **annotation,
                            },
                            index=[0],
                        )
                    )

        ## texture_features
        if annotation_type == _vars._texture_feature_type:
            for annotation_id in annotations[annotation_type].keys():
                contour_id = annotations[annotation_type][annotation_id]["settings"][_vars._contour_type + "_id"]
                for idx, annotation in enumerate(
                    annotations[annotation_type][annotation_id]["data"][annotation_type], 1
                ):
                    list_flattened.append(
                        pd.DataFrame(
                            {
                                **{"image_name": image_name},
                                **{"annotation_type": annotation_type},
                                **{"annotation_id": annotation_id},
                                **{"contour_id": contour_id},
                                **{"contour_idx": idx},
                                **annotation,
                            },
                            index=[0],
                        )
                    )

        if len(list_flattened) > 0:
            ul._print("- exported csv for type " + annotation_type)
            df = pd.concat(list_flattened)
            filepath = os.path.join(
                dir_path, save_prefix + annotation_type + save_suffix + ".csv"
            )
            df.to_csv(filepath, index=False)


def load_annotation(filepath, annotation_type=None, annotation_id=None, tag=None,**kwargs):
    """
    load phenopype annotations file

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
            try:
                annotation_file = json.load(file)
            except Exception as ex:
                ul._print(
                    "load_annotation: " + str(ex.__class__.__name__) + " - " + str(ex)
                )
                return
        annotation_file = defaultdict(dict, annotation_file)

    elif os.path.isdir(filepath):
        if tag.__class__.__name__ == "NoneType":
            ul._print("Attempting to load directory without specifying tag - aborting")
            return
        else:
            filepath = os.path.join(filepath, "annotations_{}.json".format(tag))
            with open(filepath) as file:
                annotation_file = json.load(file)
    else:
        ul._print("Annotation file not found")
        return

    ## parse serialized array
    for annotation_type1 in annotation_file:
        for annotation_id1 in annotation_file[annotation_type1]:
            for section in annotation_file[annotation_type1][annotation_id1]:
                for key, value in annotation_file[annotation_type1][annotation_id1][
                    section
                ].items():
                    if key in [
                        x
                        for x in _vars._annotation_types
                        if not x in [_vars._comment_type, _vars._reference_type,]
                    ] + ["support"]:
                        if type(value) == str:
                            value = eval(value)
                        if key == _vars._contour_type:
                            value = [np.asarray(elem, dtype=np.int32) for elem in value]
                        elif annotation_type1 == _vars._landmark_type:
                            value = [tuple(elem) for elem in value]
                    annotation_file[annotation_type1][annotation_id1][section][
                        key
                    ] = value

    ## subsetting
    while True:

        ## filter by annotation type
        if annotation_type.__class__.__name__ == "NoneType":
            ul._print("- no annotation_type selected - returning all annotations")
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
            ul._print(
                "- no annotation_id selected - returning all annotations of type: {}".format(
                    annotation_type_list
                )
            )
            annotation = annotation_subset
            break
        elif annotation_id.__class__.__name__ == "str":
            annotation_id_list = [annotation_id]
        elif annotation_id.__class__.__name__ == "list":
            annotation_id_list = annotation_id
            pass
        annotation = defaultdict(dict)
        for annotation_type in annotation_type_list:
            for annotation_id in annotation_id_list:
                if annotation_id in annotation_subset[annotation_type]:
                    annotation[annotation_type][annotation_id] = annotation_subset[
                        annotation_type
                    ][annotation_id]
        break

    ## return
    return dict(annotation)



def save_annotation(
    annotations,
    dir_path,
    annotation_type=None,
    annotation_id=None,
    overwrite=False,
    **kwargs
):
    """
    save phenopype annotations file

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
    dir_path : str, optional
        Path to folder where annotation should be saved. None will save the
        annotation in the current Python working directory. The default is None.
    file_name : str, optional
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
    file_name = kwargs.get("file_name","annotations")

    if not file_name.endswith(".json"):
        file_name = file_name + ".json"

    ## filepath
    filepath = os.path.join(dir_path, file_name)
    annotations = copy.deepcopy(annotations)

    ## open existing json or create new
    while True:
        if os.path.isfile(filepath):
            if overwrite in [False, True, "entry"]:
                with open(filepath) as file:
                    try:
                        annotation_file = json.load(file)
                        annotation_file = defaultdict(dict, annotation_file)
                    except Exception as ex:
                        ul._print(
                            "load_annotation: "
                            + str(ex.__class__.__name__)
                            + " - "
                            + str(ex)
                        )
                        ul._print("Could not read {} - creating new file.".format(filepath))
                        annotation_file = defaultdict(dict, {})

                ul._print("- loading existing annotation file")
                break
            elif overwrite == "file":
                pass
                ul._print('- overwriting annotation file (overwrite="file")')
        else:
            ul._print("- creating new annotation file")
            pass
        annotation_file = defaultdict(dict)
        break

    annotations = defaultdict(dict, annotations)

    ## subset annotation types
    if annotation_type.__class__.__name__ == "NoneType":
        ul._print("- no annotation_type selected - exporting all annotations")
        annotation_types = list(annotations.keys())
    elif annotation_type.__class__.__name__ == "str":
        annotation_types = [annotation_type]
    elif annotation_type.__class__.__name__ in ["list", "CommentedSeq"]:
        annotation_types = annotation_type

    for annotation_type in annotation_types:

    ## write annotations to output dict
    # for annotation_type in annotations:
        for annotation_id in annotations[annotation_type]:
            annotation_id_new = annotation_id
            if str(annotation_id) in annotation_file[annotation_type]:
                if overwrite in [True, "entry"]:
                    annotation_file[annotation_type][annotation_id_new] = annotations[
                        annotation_type
                    ][annotation_id]
                    ul._print(
                        '- updating annotations of type "{}" with id '
                        '"{}" in "{}" (overwrite="entry")'.format(
                            annotation_type, annotation_id, file_name
                        )
                    )
                else:
                    ul._print(
                        '- annotations of type "{}" with id "{}" already '
                        'exists in "{}" (overwrite=False)'.format(
                            annotation_type, annotation_id, file_name
                        )
                    )
            else:
                annotation_file[annotation_type][annotation_id_new] = annotations[
                    annotation_type
                ][annotation_id]
                ul._print(
                    '- writing annotations of type "{}" with id '
                    '"{}" to "{}"'.format(annotation_type, annotation_id, file_name)
                )

    ## remove indents from annotations arrays and lists
    for annotation_type in annotation_file:
        for annotation_id in annotation_file[annotation_type]:
            for section in annotation_file[annotation_type][annotation_id]:
                for key, value in annotation_file[annotation_type][annotation_id][section].items():

                    ## unindent lists for better legibility
                    if key in [x for x in _vars._annotation_types if x not in [_vars._comment_type, _vars._reference_type]] + ["support"]:
                        if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                            value = [elem.tolist() for elem in value]
                        value = [ul._NoIndent(elem) for elem in value]
                    elif isinstance(value, (tuple, list)):
                        value = ul._NoIndent(value)
                    annotation_file[annotation_type][annotation_id][section][key] = value

    ## save
    with open(filepath, "w") as file:
        json.dump(
            annotation_file, file, indent=indent, cls=ul._NoIndentEncoder
        )

def save_ROI(
    image,
    annotations,
    dir_path,
    file_name,
    channel="raw",
    counter=True,
    which="all",
    suffix=None,
    rotate=False,
    rotate_padding=5,
    angle_apply = None,
    align="v",
    ext="jpg",
    background="original",
    canvas_dim=False,
    canvas_fit=False,
    min_dim=False,
    max_dim=False,
    padding=True,
    **kwargs
    
):
    """
    
    Save a region of interest (ROI) indicated by contour or mask coordinates as
    a crop of the original image, optionally with white background
    
    Parameters
    ----------
    image : ndarray
        An image containing regions of interest (ROI).
    annotations : dict
        A phenopype annotation dict containing one or more contour coordinate
        entries.
    dir_path : str, optional
        Path to folder where annotation should be saved. None will save the
        annotation in the current Python working directory. The default is None.
    file_name : str
        Name for ROI series (should reflect image content, not "ROI" or the like
        which is specified with prefix or suffix arguments). The contour index
        will be added as a numeric string at the end of the filename.
    channel : str, optional
        Which channel to save. The default is "raw".
    counter : TYPE, optional
        Whether to add a contour to the filename. The default is True.
    prefix : str, optional
        Prefix to prepend to individual ROI filenames. The default is None.
    suffix : str, optional
        Suffix to append to individual ROI filenames. The default is "roi".
    extension : str, optional
        New e. The default is "png".
    background : str, optional
        Sets background. The default is "original", providing the background 
        contained within the bounding rectangle. "transparent" will produce a
        png file with tranparent background. "white", "black" or any other 
        color will produce a different background color.
        
    Returns
    -------
    None.

    """

    # =============================================================================
    # annotation management

    annotation_type = kwargs.get("annotation_type", _vars._mask_type)
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    data = annotation["data"][annotation_type]
    
    # =============================================================================
    # prep save
    
    ## add suffix 
    if suffix is None:
        suffix = ""
    else:
        suffix = "_" + suffix
        
    ## add extension
    if "." not in ext:
        ext = "." + ext
        
    # =============================================================================
    # run
    
    if which == "max":
        area = list()
        for idx, roi_coords in enumerate(data):
    
            if annotation_type == _vars._mask_type:
                coords = ul._convert_tup_list_arr(roi_coords)[0]
            else:
                coords = copy.deepcopy(roi_coords)
        
            area.append(int(cv2.contourArea(coords)))
        
            data = [data[area.index(max(area))]]
    
    if not channel=="raw":
        image = preprocessing.decompose_image(image, channel, **kwargs)
        
    for idx, roi_coords in enumerate(data):

        if annotation_type == _vars._mask_type and not isinstance(roi_coords, np.ndarray):
            coords = ul._convert_tup_list_arr(roi_coords)
        else:
            coords = copy.deepcopy(roi_coords)


        # =============================================================================
        ## create roi and mask
        
        if min_dim:
            (x_center, y_center), diameter = cv2.minEnclosingCircle(coords)
            half_length = min_dim // 2
            rx = int(max(0, x_center - half_length))
            ry = int(max(0, y_center - half_length))
            rh, rw = min_dim, min_dim
            roi = copy.deepcopy(image[ry : ry + rh, rx : rx + rw])
        else:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            roi = copy.deepcopy(image[ry : ry + rh, rx : rx + rw])
            half_length = 0
            
        if roi.ndim == 3:
            nlayer = 3
        else:
            nlayer = 1

        roi_mask = np.zeros(roi.shape[:2], dtype="uint8")
        roi_mask = cv2.drawContours(
            image=roi_mask,
            contours=[coords],
            contourIdx=0,
            thickness=-1,
            color=255,
            offset=(-rx, -ry),
        )
        
        # =============================================================================
        ## rotate roi and mask
        
        if rotate:
            if angle_apply == None:
                angle = ul._get_orientation(coords)
                if align == "h":
                    angle = angle + 90
                elif align == "v":
                    angle = angle - 90
            else:
                angle = angle_apply
            roi_rot = ul._rotate_image(roi, angle, allow_crop=False)
            roi_mask_rot = ul._rotate_image(roi_mask, angle, allow_crop=False)
            contour, _ = cv2.findContours(roi_mask_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rx, ry, rw, rh = cv2.boundingRect(contour[0])
            roi = roi_rot[ry : ry + rh, rx : rx + rw]
            roi_mask = roi_mask_rot[ry : ry + rh, rx : rx + rw]
            
            ## pad to avoid aliasing
            roi = cv2.copyMakeBorder(roi, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)    
            roi_mask = cv2.copyMakeBorder(roi_mask, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)  
            
        # =============================================================================
        ## mount on canvas
        
        if canvas_dim:
            
            ## make canvas
            roi_canvas = np.zeros((canvas_dim, canvas_dim, nlayer), dtype=np.uint8)
            roi_mask_canvas = np.zeros((canvas_dim, canvas_dim), dtype=np.uint8)

            # calculate the new top-left coordinates for the smaller ROI
            xpos = (canvas_dim - roi.shape[1]) // 2
            ypos = (canvas_dim - roi.shape[0]) // 2
            
            ## skip if ROI is too large for canvas
            if any([xpos<=0, ypos<=0]):
                if canvas_fit:
                    roi = utils.resize_image(roi, max_dim=canvas_dim-2)
                    roi_mask = utils.resize_image(roi_mask, max_dim=canvas_dim-2)
                    xpos = max((canvas_dim - roi.shape[1]) // 2,1)
                    ypos = max((canvas_dim - roi.shape[0]) // 2,1)
                else:
                    ul._print(f"ROI {idx+1} too big for chosen canvas: {roi.shape[:2]} vs {(canvas_dim,canvas_dim)}. Skipping!", lvl=1)
                    continue
            
            # Place the smaller ROI onto the larger ROI canvas
            roi_canvas[ypos:ypos + roi.shape[0], xpos:xpos + roi.shape[1]] = roi
            roi_mask_canvas[ypos:ypos + roi_mask.shape[0], xpos:xpos + roi_mask.shape[1]] = roi_mask
            
            ## replace original
            roi,roi_mask  = roi_canvas, roi_mask_canvas
            
        # =============================================================================
        ## background handling
        
        if background=="original":           
            pass
        else:
            if background=="transparent":           
                roi_alpha = np.zeros((roi.shape[0], roi.shape[1], 4), dtype=np.uint8)
                roi_alpha[:,:,0:3] = roi
                roi_alpha[:, :, 3] = roi_mask
                roi_canvas = roi_alpha
                ext = ".png"
            else:
                roi[roi_mask==0] = ul._get_bgr(background)
                roi_canvas = roi
            roi = roi_canvas
                
                
        # =============================================================================
        ## resizing final ROI
        
        if max_dim:
            roi = utils.resize_image(roi, max_dim=max_dim)
            
        # =============================================================================
        ## saving          
                
        if not kwargs.get("training_data"):
                             
            ## add counter
            if counter:
                roi_name = file_name + suffix + "_" + str(idx+1).zfill(3) + ext
            else:
                roi_name = file_name + suffix + ext  
            
            save_path = os.path.join(dir_path, roi_name)
            saved = cv2.imwrite(save_path, roi) 
            
            if saved:
                ul._print("- saving ROI: {}".format(roi_name))
            else:
                ul._print("- something went wrong - didn't save ROI")
                
        else:
            return roi, roi_mask
        
@decorators.legacy_args
def save_canvas(
        image, 
        file_path, 
        ext="jpg",
        resize=0.5,
        overwrite=True,
        **kwargs,
        ):
    """


    Parameters
    ----------
    image : ndarray
        A canvas to be saved.
    save_suffix : str
        A suffix to be appended to the filename.
    dir_path : str
        Path to directory to save canvas.
    resize: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of

    Returns
    -------
    None.

    """
    
    image = utils.resize_image(image, resize)

    utils.save_image(
        image=image,
        file_path=file_path,
        ext=ext,
        overwrite=overwrite,
        **kwargs
    )
