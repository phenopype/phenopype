#%% modules

import copy
import cv2
import numpy as np
import sys

from dataclasses import make_dataclass

from phenopype import __version__
from phenopype import _config
from phenopype import plugins
from phenopype import settings
from phenopype import utils_lowlevel
from phenopype.core import segmentation, visualization
from phenopype.utils_lowlevel import annotation_function


#%% functions

def detect_object(
    image,
    model_path,
    model_id=None,
    binary_mask=False,
    threshold=True,
    threshold_method="otsu",
    threshold_value=127,
    threshold_blocksize=99,
    threshold_constant=5,
    force_reload=False,
    **kwargs,
):
    """
    Applies a trained deep learning model to an image and returns a grayscale mask 
    of foreground predictions, which can then be thresholded to return a binary mask.
    
    Three types of thresholding algorithms are supported: 
        - otsu: use Otsu algorithm to choose the optimal threshold value
        - adaptive: dynamic threshold values across image (uses arguments
          "blocksize" and "constant")
        - binary: fixed threshold value (uses argument "value")    
        
    Parameters
    ----------
    image : ndarray
        input image
    model_path : str
        path to a detection model (currently only keras h5 objects are supported)
    model_id : str, optional
        id for a model that has been added to a phenopype project (overrides model_path)
    threshold : bool, optional
        perform thresholding on returned grayscale segmentation mask to create binary image.
        default is True.
    threshold_method : {"otsu", "adaptive", "binary"} str, optional
        type of thresholding algorithm to be used on the model output
    threshold_blocksize: int, optional
        Size of a pixel neighborhood that is used to calculate a threshold 
        value for the model mask (has to be odd - even numbers will be ceiled; for
        "adaptive" method)
    threshold_constant : int, optional
        value to subtract from binarization output (for "adaptive" method)
    threshold_value : {between 0 and 255} int, optional
        thesholding value (for "binary" method)
    force_reload : bool, optional
        force a model reload every time the function is run (WARNING: this may 
        take a long time)     

    Returns
    -------
    image : ndarray
        binary image

    """
    # =============================================================================
    # setup
    
    fun_name = sys._getframe().f_code.co_name
    
    ## flags
    flags = make_dataclass(cls_name="flags", 
                           fields=[("binary_mask", bool, binary_mask)])
    
    # =============================================================================
    # annotation management
    if flags.binary_mask:
        
        annotations = kwargs.get("annotations", {})
        annotation_type = kwargs.get("annotation_type", settings._mask_type)
        annotation_id = kwargs.get(annotation_type + "_id", None)
            
    # =============================================================================
    # execute
    
    image_source = copy.deepcopy(image)
    
    if flags.binary_mask:
        binary_mask = np.zeros(image_source.shape, dtype="uint8")
        if annotation_type == settings._mask_type:
            print("mask")
            binary_mask = visualization.draw_mask(
                image=binary_mask, 
                annotations=annotations, 
                contour_id=annotation_id, 
                line_colour=255,
                line_width=0,
                fill=1)
        elif annotation_type == settings._contour_type:
            print("contour")
            binary_mask = visualization.draw_contour(
                image=binary_mask, 
                annotations=annotations, 
                contour_id=annotation_id, 
                line_colour=255,
                line_width=0,
                fill=1)

        image_source = cv2.bitwise_and(image_source, binary_mask)

    if model_path.__class__.__name__ == "NoneType":
        print("No model provided - did you set an active model?")
        return image_source
    
    if not model_id.__class__.__name__ == "NoneType":
        model_path = _config.models[model_id]["model_phenopype_path"]
        if not "model_loaded" in _config.models[model_id]:
            print("loading model " + model_id)
            _config.models[model_id]["model_loaded"] = plugins.libraries.keras.models.load_model(model_path)
        _config.active_model = _config.models[model_id]["model_loaded"]
        _config.active_model_path = model_path
    
    elif not _config.active_model_path == model_path or _config.active_model.__class__.__name__ == "NoneType" or force_reload==True:
        _config.active_model = plugins.libraries.keras.models.load_model(model_path)
        _config.active_model_path = model_path

    print("Using current model at " + _config.active_model_path)
    
    model = _config.active_model

    image_source = utils_lowlevel._resize_image(image_source, width=model.input.shape[1], height=model.input.shape[2])/255
    image_source = np.expand_dims(image_source, axis=0)
    pred = model.predict(image_source)
     
    mask_predicted = pred[0,:,:,1]*255
    mask_predicted = mask_predicted.astype(np.uint8)
    mask_predicted = utils_lowlevel._resize_image(mask_predicted, width=image.shape[1], height=image.shape[0], interpolation="linear")
    
    if threshold:
        mask_predicted = segmentation.threshold(
            mask_predicted, 
            invert=True,
            method=threshold_method,
            value=threshold_value, 
            blocksize=threshold_blocksize,
            constant=threshold_constant
            )
           

    # tf.keras.backend.clear_session()
    
    return mask_predicted
