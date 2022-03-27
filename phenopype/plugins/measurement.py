#%% imports

clean_namespace = dir()

import cv2
import math
import numpy as np
import sys

from phenopype import __version__

from phenopype import plugins
from phenopype import settings
from phenopype import utils_lowlevel


#%% namespace cleanup

funs = ['detect_landmark']

def __dir__():
    return clean_namespace + funs

#%% functions

def detect_landmark(
    image,
    point_colour="default",
    point_size="auto",
    label=True,
    label_colour="default",
    label_size="auto",
    label_width="auto",
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

    # annotations = kwargs.get("annotations", {})
    # annotation_type = utils_lowlevel._get_annotation_type(fun_name)
    # annotation_id = kwargs.get("annotation_id", None)

    # annotation = utils_lowlevel._get_annotation(
    #     annotations=annotations,
    #     annotation_type=annotation_type,
    #     annotation_id=annotation_id,
    #     kwargs=kwargs,
    # )

    # gui_data = {settings._coord_type: utils_lowlevel._get_GUI_data(annotation)}
    # gui_settings = utils_lowlevel._get_GUI_settings(kwargs, annotation)
    

    # =============================================================================
    # check dependencies
    
    if hasattr(plugins, "phenomorph"):
        from phenopype.plugins import phenomorph
    else:
        print("can't use {} - phenomorph not loaded".format(fun_name))
        return
    
    phenomorph.model.Model.predict_image(tag='lm-v1', img_path='./phenopype/CxG_F2/training_data/ml-morph-crops/images/stickle001_004.JPG')

    
    
    # try:
    #     from phenomorph.model import Model
    # except:    
    #     print("can't use {} - phenomorph not loaded".format("detect_lm"))
    #     return
    
    # print(Model)

    # =============================================================================
    # further prep

    # ## configure points
    # if point_size == "auto":
    #     point_size = utils_lowlevel._auto_point_size(image)
    # if label_size == "auto":
    #     label_size = utils_lowlevel._auto_text_size(image)
    # if label_width == "auto":
    #     label_width = utils_lowlevel._auto_text_width(image)
    # if label_colour == "default":
    #     label_colour = settings._default_label_colour
    # if point_colour == "default":
    #     point_colour = settings._default_point_colour

    # label_colour = utils_lowlevel._get_bgr(label_colour)
    # point_colour = utils_lowlevel._get_bgr(point_colour)

    # # =============================================================================
    # # execute

    # gui = utils_lowlevel._GUI(
    #     image=image,
    #     tool="point",
    #     label=label,
    #     point_size=point_size,
    #     point_colour=point_colour,
    #     label_size=label_size,
    #     label_width=label_width,
    #     label_colour=label_colour,
    #     data=gui_data,
    #     **gui_settings,
    # )

    # # =============================================================================
    # # assemble results

    # annotation = {
    #     "info": {
    #         "annotation_type": annotation_type,
    #         "phenopype_function": fun_name,
    #         "phenopype_version": __version__,
    #     },
    #     "settings": {
    #         "point_size": point_size,
    #         "point_colour": point_colour,
    #         "label": label,
    #         "label_size": label_size,
    #         "label_width": label_width,
    #         "label_colour": label_colour,
    #     },
    #     "data": {annotation_type: gui.data[settings._coord_type],},
    # }

    # if len(gui_settings) > 0:
    #     annotation["settings"]["GUI"] = gui_settings

    # # =============================================================================
    # # return

    # return utils_lowlevel._update_annotations(
    #     annotations=annotations,
    #     annotation=annotation,
    #     annotation_type=annotation_type,
    #     annotation_id=annotation_id,
    #     kwargs=kwargs,
    # )


