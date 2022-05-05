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

if hasattr(plugins, "ml_morph"):
    import phenomorph
    import dlib

#%% namespace cleanup

funs = ['detect_landmark']

def __dir__():
    return clean_namespace + funs

#%% functions

def detect_landmark(
    image,
    model_id=None,
    model_folder=None,
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
    
    ## initate model
    model = phenomorph.model.Model(rootdir=model_folder)
  
    if kwargs.get("tag") and not model_id:
        model_id = kwargs.get("tag")
    
    if mask:        
        if not annotations:
            print("- no mask coordinates provided - cannot detect within mask")
            pass
        else:
            annotation_id_mask = kwargs.get(settings._mask_type + "_id", None)
            annotation_mask = utils_lowlevel._get_annotation(
                annotations,
                settings._mask_type,
                annotation_id_mask,
                prep_msg="- masking regions in thresholded image:",
            )
            
            bbox_coords = cv2.boundingRect(np.asarray(annotation_mask["data"][settings._mask_type], dtype="int32"))
            landmark_tuple_list = model.predict_image(tag=model_id, img=image,  bbox_coords = bbox_coords, plot=False)
    else:
        landmark_tuple_list = model.predict_image(tag=model_id, img=image, plot=False)


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
        
