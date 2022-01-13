#%% modules
import copy
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


from phenopype import settings
from phenopype import utils_lowlevel
from phenopype import utils

#%% functions


def export_csv(
    annotations,
    dir_path,
    annotation_type=None,
    image_name=None,
    overwrite=False,
    **kwargs
):
    """
    

    Parameters
    ----------
    annotations : TYPE
        DESCRIPTION.
    annotation_type : TYPE, optional
        DESCRIPTION. The default is None.
    image_name : TYPE, optional
        DESCRIPTION. The default is None.
    dir_path : TYPE, optional
        DESCRIPTION. The default is None.
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    ## dirpath

    if not image_name:
        print(
            "Warning: missing image_name argument - exported CSV will not contain information about source-image"
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
        print("- no annotation_type selected - exporting all annotations")
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
        if annotation_type == settings._comment_type:
            for annotation_id in annotations[annotation_type].keys():
                list_flattened.append(
                    pd.DataFrame(
                        {
                            **{"image_name": image_name},
                            **{"annotation_type": annotation_type},
                            **{"annotation_id": annotation_id},
                            **{
                                "label": annotations[annotation_type][annotation_id][
                                    "data"
                                ][annotation_type]
                            },
                        },
                        index=[0],
                    )
                )

        ## contour
        if annotation_type == settings._contour_type:
            for annotation_id in annotations[annotation_type].keys():
                idx = 0
                for coords, support in zip(
                    annotations[annotation_type][annotation_id]["data"][
                        annotation_type
                    ],
                    annotations[annotation_type][annotation_id]["data"]["support"],
                ):
                    idx += 1
                    list_flattened.append(
                        # df_temp = pd.DataFrame(utils_lowlevel._convert_arr_tup_list(coords)[0], columns=["x","y"])
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
                                **{
                                    "hierarchy_idx_child": support[
                                        "hierarchy_idx_child"
                                    ]
                                },
                                **{
                                    "hierarchy_idx_parent": support[
                                        "hierarchy_idx_parent"
                                    ]
                                },
                            },
                            index=[0],
                        )
                    )

        ## shape_features
        if annotation_type == settings._shape_feature_type:
            for annotation_id in annotations[annotation_type].keys():
                for idx, annotation in enumerate(
                    annotations[annotation_type][annotation_id]["data"][annotation_type]
                ):
                    list_flattened.append(
                        pd.DataFrame(
                            {
                                **{"image_name": image_name},
                                **{"annotation_type": annotation_type},
                                **{"annotation_id": annotation_id},
                                **{"contour_idx": idx},
                                **annotation,
                            },
                            index=[0],
                        )
                    )

        ## texture_features
        if annotation_type == settings._texture_feature_type:
            for annotation_id in annotations[annotation_type].keys():
                for idx, annotation in enumerate(
                    annotations[annotation_type][annotation_id]["data"][annotation_type]
                ):
                    list_flattened.append(
                        pd.DataFrame(
                            {
                                **{"image_name": image_name},
                                **{"annotation_type": annotation_type},
                                **{"annotation_id": annotation_id},
                                **{"contour_idx": idx},
                                **annotation,
                            },
                            index=[0],
                        )
                    )

        if len(list_flattened) > 0:
            print("- exported csv for type " + annotation_type)
            df = pd.concat(list_flattened)
            filepath = os.path.join(
                dir_path, save_prefix + annotation_type + save_suffix + ".csv"
            )
            df.to_csv(filepath, index=False)


def load_annotation(filepath, annotation_type=None, annotation_id=None):
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
            try:
                annotation_file = json.load(file)
            except Exception as ex:
                print(
                    "load_annotation: " + str(ex.__class__.__name__) + " - " + str(ex)
                )
                return
        annotation_file = defaultdict(dict, annotation_file)
    else:
        print("file not found")
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
                        for x in settings._annotation_types
                        if not x in [settings._comment_type, settings._reference_type,]
                    ] + ["support"]:
                        if type(value) == str:
                            value = eval(value)
                        if key == settings._contour_type:
                            value = [np.asarray(elem, dtype=np.int32) for elem in value]
                        elif annotation_type1 == settings._landmark_type:
                            value = [tuple(elem) for elem in value]
                    annotation_file[annotation_type1][annotation_id1][section][
                        key
                    ] = value

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
            print(
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
    annotation,
    dir_path,
    file_name="annotations.json",
    annotation_id=None,
    overwrite=False,
    **kwargs
):
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

    if not file_name.endswith(".json"):
        file_name = file_name + ".json"

    ## filepath
    filepath = os.path.join(dir_path, file_name)
    annotation = copy.deepcopy(annotation)

    ## open existing json or create new
    while True:
        if os.path.isfile(filepath):
            if overwrite in [False, True, "entry"]:
                with open(filepath) as file:
                    try:
                        annotation_file = json.load(file)
                        annotation_file = defaultdict(dict, annotation_file)
                    except Exception as ex:
                        print(
                            "load_annotation: "
                            + str(ex.__class__.__name__)
                            + " - "
                            + str(ex)
                        )
                        print("Could not read {} - creating new file.".format(filepath))
                        annotation_file = defaultdict(dict, {})

                print("- loading existing annotation file")
                break
            elif overwrite == "file":
                pass
                print('- overwriting annotation file (overwrite="file")')
        else:
            print("- creating new annotation file")
            pass
        annotation_file = defaultdict(dict)
        break

    annotation = defaultdict(dict, annotation)

    ## write annotation to output dict
    for annotation_type in annotation:
        for annotation_id in annotation[annotation_type]:
            annotation_id_new = annotation_id
            if str(annotation_id) in annotation_file[annotation_type]:
                if overwrite in [True, "entry"]:
                    annotation_file[annotation_type][annotation_id_new] = annotation[
                        annotation_type
                    ][annotation_id]
                    print(
                        '- updating annotation of type "{}" with id '
                        '"{}" in "{}" (overwrite="entry")'.format(
                            annotation_type, annotation_id, file_name
                        )
                    )
                else:
                    print(
                        '- annotation of type "{}" with id "{}" already '
                        'exists in "{}" (overwrite=False)'.format(
                            annotation_type, annotation_id, file_name
                        )
                    )
            else:
                annotation_file[annotation_type][annotation_id_new] = annotation[
                    annotation_type
                ][annotation_id]
                print(
                    '- writing annotation of type "{}" with id '
                    '"{}" to "{}"'.format(annotation_type, annotation_id, file_name)
                )

    ## remove indents from annotation arrays and lists
    for annotation_type in annotation_file:
        for annotation_id in annotation_file[annotation_type]:
            for section in annotation_file[annotation_type][annotation_id]:
                for key, value in annotation_file[annotation_type][annotation_id][
                    section
                ].items():

                    ## unindent lists for better legibility
                    if key in [
                        x
                        for x in settings._annotation_types
                        if not x in [settings._comment_type, settings._reference_type,]
                    ] + ["support"]:
                        if (
                            type(value) == list
                            and len(value) > 0
                            and type(value[0]) in [np.ndarray]
                        ):
                            value = [elem.tolist() for elem in value]
                        value = [utils_lowlevel._NoIndent(elem) for elem in value]
                    elif type(value) in [tuple, list]:
                        value = utils_lowlevel._NoIndent(value)

                    annotation_file[annotation_type][annotation_id][section][
                        key
                    ] = value

    ## save
    with open(filepath, "w") as file:
        json.dump(
            annotation_file, file, indent=indent, cls=utils_lowlevel._NoIndentEncoder
        )


def save_ROI(
    image, annotations, dir_path, file_name, prefix=None, suffix="roi", **kwargs
):
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
    dir_path : str, optional
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

    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + "_"
    if suffix is None:
        suffix = ""
    else:
        suffix = "_" + suffix

    # =============================================================================
    # annotation management

    annotation_type = settings._contour_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    contours = annotation["data"][annotation_type]

    for idx, roi_coords in enumerate(contours):

        rx, ry, rw, rh = cv2.boundingRect(roi_coords)
        roi_rect = image[ry : ry + rh, rx : rx + rw]

        roi_name = prefix + file_name + suffix + "_" + str(idx).zfill(2) + ".tif"

        save_path = os.path.join(dir_path, roi_name)
        cv2.imwrite(save_path, roi_rect)

        # roi_new_coords = []
        # for coord in roi_coords:
        #     new_coord = [coord[0][0] - rx, coord[0][1] - ry]
        #     roi_new_coords.append([new_coord])
        # roi_new_coords = np.asarray(roi_new_coords, np.int32)


def save_canvas(image, dir_path, file_name="canvas", **kwargs):
    """
    

    Parameters
    ----------
    image : ndarray
        A canvas to be saved.
    save_suffix : str
        A suffix to be appended to the filename.
    dir_path : str
        Path to directory to save canvas.

    Returns
    -------
    None.

    """

    ext = kwargs.get("ext", ".jpg")
    resize = kwargs.get("resize", 1)
    overwrite = kwargs.get("overwrite", True)

    utils.save_image(
        image=image,
        file_name=file_name,
        ext=ext,
        dir_path=dir_path,
        resize=resize,
        overwrite=overwrite,
        verbose=settings.flag_verbose,
    )
