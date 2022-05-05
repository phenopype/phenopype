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

if hasattr(plugins, "keras_cnn"):
    import tensorflow as tf


#%% functions


def detect_object(
    image,
    model,
    **kwargs,
):



    model = tf.keras.models.load_model(r"D:\science\projects\2021_odonata_scans\jupyter\cnn1\UNet_resnet50_v1_IS512_EP100_BS16.h5")
    
    _config.current_model = model
    
    model.input.shape
    
    image = np.expand_dims(image, axis=0)
    
    pred = model.predict(image)

    

mo