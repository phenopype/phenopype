#%% imports

clean_namespace = dir()

import cv2
import math
import numpy as np
import os
import sys

from phenopype import __version__

from phenopype import plugins
from phenopype import settings
from phenopype import utils
from phenopype import utils_lowlevel

import phenomorph as ml_morph
import dlib

# if hasattr(plugins, "phenomorph"):
#     from phenopype.plugins import phenomorph
# else:
#     print("can't use {} - phenomorph not loaded".format(fun_name))
#     return

#%% namespace cleanup

funs = ['detect_landmark']

def __dir__():
    return clean_namespace + funs

#%% functions

def detect_landmark(
    image,
    model_folder=None,
    model_tag=None,
    mask=True,
    **kwargs,
):
    """
    Place landmarks. Note that modifying the appearance of the points will only 
    be effective for the placement, not for subsequent drawing, visualization, 
    and export.
    
    Parameters
    ----------
    image : ndarray
        input image
    point_colour: str, optional
        landmark point colour (for options see pp.colour)
    point_size: int, optional
        landmark point size in pixels
    label : bool, optional
        add text label
    label_colour : str, optional
        landmark label colour (for options see pp.colour)
    label_size: int, optional
        landmark label font size (scaled to image)
    label_width: int, optional
        landmark label font width  (scaled to image)

    Returns
    -------
    annotations: dict
        phenopype annotation containing landmarks
    """

    # =============================================================================
    # annotation management

    fun_name = sys._getframe().f_code.co_name
    
    annotations = kwargs.get("annotations", {})
    annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
    
    
    landmark_tuple_list = []
    
    rect = dlib.rectangle(
        left=1, top=1, right=image.shape[1] - 1, bottom=image.shape[0] - 1
    )
    
    if mask:        
        if not annotations:
            print("no mask coordinates provided - cannot detect within mask")
            pass
        else:
            annotation_id_mask = kwargs.get(settings._mask_type + "_id", None)
            annotation_mask = utils_lowlevel._get_annotation(
                annotations,
                settings._mask_type,
                annotation_id_mask,
                prep_msg="- masking regions in thresholded image:",
            )
            
            rx, ry, rw, rh = cv2.boundingRect(np.asarray(annotation_mask["data"][settings._mask_type], dtype="int32"))
            rect = dlib.rectangle(
                left=rx, top=ry, right=rx+rw, bottom=ry+rh
            )
        
    ## prediction
    predictor = dlib.shape_predictor(os.path.join(model_folder, "predictor_{}.dat".format(model_tag)))   
    predicted_points = predictor(image, rect)

    ## wrong order
    points_dict = {}
    for point, idx in enumerate(sorted(range(0, predicted_points.num_parts), key=str)):
        x,y = predicted_points.part(point).x, predicted_points.part(point).y
        points_dict[idx] = (x,y)
        
    ## correct order
    landmark_tuple_list = []
    for key, value in sorted(points_dict.items()): 
        landmark_tuple_list.append(value)
        
    print("- using ml-morph landmark predictor: found {} points".format(len(landmark_tuple_list)))

    annotation = {
        "info": {
            "annotation_type": annotation_type,
            "phenopype_function": "plugins.ml_morph.predict_image",
            "phenopype_version": __version__,
        },
        "settings": {
        },
        "data": {
            annotation_type: landmark_tuple_list
            },
    }


    return utils_lowlevel._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
        
