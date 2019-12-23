#%% modules
import cv2
import copy
import math
import numpy as np

from phenopype.settings import colours
from phenopype.utils_lowlevel import _auto_line_thickness

from phenopype.preprocessing import show_mask

#%% settings

inf = math.inf

#%% methods