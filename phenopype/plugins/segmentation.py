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

from phenopype import utils

#%% functions

def convert_box_xywh_to_xyxy(box):
    if len(box) == 4:
        return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    else:
        result = []
        for b in box:
            b = convert_box_xywh_to_xyxy(b)
            result.append(b)               
    return result

def predict_SAM(
        image,
        model_path,
        model_id=None,
        prompt="local",
        prompt_type="box",
        resize_roi=1024,
        center=0.9,
        confidence=0.8,
        iou=0.65,
        force_reload=False,
        **kwargs,
        ):
        
    # =============================================================================
    # setup
        
    ## set flags
    flags = make_dataclass(
        cls_name="flags",
        fields=[("prompt", str, prompt), 
                ("max_dim", str, kwargs.get("max_dim")), 
                ],
    )
    
    # =============================================================================
    # model management

    if model_path.__class__.__name__ == "NoneType":
        print("No model provided - did you set an active model?")
        return image
    
    if not model_id.__class__.__name__ == "NoneType":
        model_path = _config.models[model_id]["model_phenopype_path"]
        if not "model_loaded" in _config.models[model_id]:
            print("loading model " + model_id)
            _config.models[model_id]["model_loaded"] = plugins.libraries.fastsam.FastSAM(model_path)
        _config.active_model = _config.models[model_id]["model_loaded"]
        _config.active_model_path = model_path
    
    elif not _config.active_model_path == model_path or _config.active_model.__class__.__name__ == "NoneType" or force_reload==True:
        _config.active_model = plugins.libraries.fastsam.FastSAM(model_path)
        _config.active_model_path = model_path

    print("Using current model at " + _config.active_model_path)
    
    ## init model and device
    device = plugins.libraries.torch.device("cuda" if plugins.libraries.torch.cuda.is_available() else "cpu")
    model = _config.active_model
                            
    # =============================================================================
    # execute
    
    if prompt_type == "box":

        ## local masking
        if flags.prompt == "local":
            
            ## masks        
            annotations = kwargs.get("annotations", {})
            annotation_id_mask = kwargs.get(settings._mask_type + "_id", None)
            annotation_mask = utils_lowlevel._get_annotation(
                annotations,
                settings._mask_type,
                annotation_id_mask,
                prep_msg="- masking regions in thresholded image:",
            )
        
            ## mask coords to roi
            coords = annotation_mask["data"]["mask"]
            coords = utils_lowlevel._convert_tup_list_arr(coords)
            rx, ry, rw, rh = cv2.boundingRect(coords[0])  
            
            ## roi
            roi_orig = image[ry : ry + rh, rx : rx + rw]
            
        ## image centered
        elif flags.prompt == "global-center":
                
            ## crop edge of image
            height, width = image.shape[:2]
            rx, ry = int(round((1 - center) * 0.5 * width)), int(round((1 - center) * 0.5 * height))
            rh, rw = int(round(center * height)), int(round(center * width))
    
            ## roi
            roi_orig = image[ry : ry + rh, rx : rx + rw]
            
        ## full image
        elif flags.prompt == "global":
            
            ## roi
            roi_orig = copy.deepcopy(image)
    
        ## resize roi
        if not resize_roi.__class__.__name__ == "NoneType":
            roi = utils_lowlevel._resize_image(
                roi_orig, width=resize_roi, height=resize_roi)
        else:
            roi = roi_orig 
            
        ## encode roi 
        everything_results = model(
            roi,
            device=device,
            retina_masks=True,
            imgsz=[int(roi.shape[1]), int(roi.shape[0])],
            conf=confidence,
            iou=iou,
        )
        
        ## box prep
        box_prompt = [[0, 0, int(roi.shape[1]), int(roi.shape[0])]]
        box_prompt = convert_box_xywh_to_xyxy(box_prompt)

        ## box prompt
        prompt_process = plugins.libraries.fastsam.FastSAMPrompt(
            roi, everything_results, device=device)
        detections = prompt_process.box_prompt(bboxes=box_prompt)
        
        # check if something was detected
        if len(detections) > 0:  
            print(f"found {len(detections)} objects!")
        else:  # Check if ann is empty
            print("SAM: no object found - moving on")
            return image
        
        # Extract mask conver to 8bit binary
        mask = (detections[0] * 255).astype(np.uint8)
    
        ## resize back to original roi
        mask_roi_orig = utils_lowlevel._resize_image(
            mask, width=roi_orig.shape[1], height=roi_orig.shape[0])
    
        ## paste roi into full mask
        mask_predicted = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_predicted[ry : ry + rh, rx : rx + rw] = mask_roi_orig
            
    return mask_predicted


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
