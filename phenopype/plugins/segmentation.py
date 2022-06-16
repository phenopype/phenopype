#%% modules

import copy
import cv2
import numpy as np
import sys

from phenopype import __version__
from phenopype import _config
from phenopype import plugins
from phenopype import settings
from phenopype import utils_lowlevel
from phenopype.core import segmentation
from phenopype.utils_lowlevel import annotation_function


#%% functions

def detect_object(
    image,
    model_path,
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
    # import checks
    
    fun_name = sys._getframe().f_code.co_name

    # =============================================================================
    # execute
    
    image_source = copy.deepcopy(image)
         
    if model_path.__class__.__name__ == "NoneType":
        print("No model provided - did you set an active model?")
        return image
    
    if not _config.current_model_path == model_path or _config.active_model.__class__.__name__ == "NoneType" or force_reload==True:
        _config.current_model_path = model_path
        _config.active_model = plugins.libraries.tensorflow.keras.models.load_model(model_path)
        print("==> loading model <==")
        
    print("Using current model at " + _config.current_model_path)
    
    model = _config.active_model

    image_source = utils_lowlevel._resize_image(image, width=model.input.shape[1], height=model.input.shape[2])/255
    image_source = np.expand_dims(image_source, axis=0)
    pred = model.predict(image_source)
     
    mask_predicted = pred[0,:,:,1]*255
    mask_predicted = mask_predicted.astype(np.uint8)
    mask_predicted = utils_lowlevel._resize_image(mask_predicted, width=image.shape[1], height=image.shape[0], interpolation="linear")
    
    mask_predicted = segmentation.threshold(
        mask_predicted, 
        invert=True,
        method=threshold_method,
        value=threshold_value, 
        blocksize=threshold_blocksize,
        constant=threshold_constant
        )
       
    return mask_predicted
