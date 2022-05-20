#%% modules

import copy
import cv2
import numpy as np
import sys

from math import inf

from phenopype import __version__
from phenopype import _config
from phenopype import plugins
from phenopype import settings
from phenopype import utils_lowlevel
from phenopype.core import preprocessing
from phenopype.utils_lowlevel import annotation_function

if hasattr(plugins, "keras_cnn"):
    import tensorflow as tf


#%% functions

# @annotation_function
def detect_object(
    image,
    model_path,
    **kwargs,
):
    
    
    # # =============================================================================
    # # annotation management

    # fun_name = sys._getframe().f_code.co_name
    # annotation_type = utils_lowlevel._get_annotation_type(fun_name)

    # annotation = kwargs.get("annotation")

    # gui_data = {settings._coord_type: utils_lowlevel._get_GUI_data(annotation)}
    # gui_settings = utils_lowlevel._get_GUI_settings(kwargs, annotation)
    
    # =============================================================================
    # execute
    
    source_image = copy.deepcopy(image)

    _config.current_model_path = model_path
    if not _config.current_model_path == _config.current_model_path and not _config.active_model.__class__.__name__ == "NoneType":
        _config.active_model = tf.keras.models.load_model(model_path)
    
    model = _config.active_model
        
    source_image = utils_lowlevel._resize_image(source_image, width=model.input.shape[1], height=model.input.shape[2])
    
    source_image = np.expand_dims(source_image, axis=0)

    pred = _config.active_model.predict(source_image)

    out = pred[0][:,:,1]
    
    # norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    
    out*=255
    out = out.astype(np.uint8)
    
    
    print("keras detection successful")
    
    return out
    
    # annotation = {
    #     "info": {
    #         "phenopype_function": fun_name,
    #         "phenopype_version": __version__,
    #         "annotation_type": annotation_type,
    #     },
    #     "settings": {
    #         "approximation": approximation,
    #         "retrieval": retrieval,
    #         "offset_coords": offset_coords,
    #         "min_nodes": min_nodes,
    #         "max_nodes": max_nodes,
    #         "min_area": min_area,
    #         "max_area": max_area,
    #         "min_diameter": min_diameter,
    #         "max_diameter": max_diameter,
    #     },
    #     "data": {
    #         "n": len(contours), 
    #         annotation_type: contours, 
    #         "support": support,},
    # }

    # # =============================================================================
    # # return

    # return utils_lowlevel._update_annotations(
    #     annotations=annotations,
    #     annotation=annotation,
    #     annotation_type=annotation_type,
    #     annotation_id=annotation_id,
    #     kwargs=kwargs,
    # )