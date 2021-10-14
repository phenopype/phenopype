#%% load modules

import cv2
import os
from importlib.resources import path
from pathlib import Path
from pprint import PrettyPrinter

import phenopype.utils_lowlevel as utils_lowlevel

#%% helper-class

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

#%% scalars and definitions

auto_line_width_factor = 0.002
auto_point_size_factor = 0.002
auto_text_width_factor = 0.0005
auto_text_size_factor = 0.00025

colours = {
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

default_filetypes = ["jpg", "JPG", "jpeg", "JPEG", "tif", "png", "bmp"]
default_meta_data_fields = [
    "DateTimeOriginal",
    "Model",
    "LensModel",
    "ExposureTime",
    "ISOSpeedRatings",
    "FNumber",
]

default_save_suffix = "v1"
default_window_size = 1000

pandas_max_rows = 10

confirm_options = ["True", "true", "y", "yes"]


#%% pype templates

default_pype_config_name = "v1"
default_pype_config = "ex1"
        
## create template list
pype_config_template_list = {}
with path(__package__, 'templates') as template_dir:
    for folder in os.listdir(template_dir):
        if not folder in ['__init__.py','__pycache__']:
            pype_config_template_list[folder] = {}
            for file in os.listdir(Path(template_dir) / folder):
                if file.endswith(".yaml"):
                    template_name = os.path.splitext(file)[0]
                    pype_config_template_list[folder][template_name] = utils_lowlevel._load_yaml(str(Path(template_dir) / folder / file))
            pype_config_template_list[folder] = AttrDict(pype_config_template_list[folder])            
pype_templates = AttrDict(pype_config_template_list)

#%% flags

flag_verbose = True

opencv_window_flags={
    "normal": cv2.WINDOW_NORMAL,
    "auto": cv2.WINDOW_AUTOSIZE,
    "openGL": cv2.WINDOW_OPENGL,
    "full": cv2.WINDOW_FULLSCREEN, 
    "free": cv2.WINDOW_FREERATIO,
    "keep": cv2.WINDOW_KEEPRATIO,
    "GUIexp": cv2.WINDOW_GUI_EXPANDED, 
    "GUInorm": cv2.WINDOW_GUI_NORMAL,
    } 

opencv_interpolation_flags = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4, 
    "lin_exact": cv2.INTER_LINEAR_EXACT, 
    "inter": cv2.INTER_MAX,
    "warp_fill": cv2.WARP_FILL_OUTLIERS,
    "warp_inverse": cv2.WARP_INVERSE_MAP, 
    }


#%% image viewer

_image_viewer_arg_list = [
    "blend_factor",
    "label_size",
    "label_width",
    "label_colour",
    "left_colour",
    "right_colour",
    "line_colour",
    "line_width",
    "passive",
    "point_size",
    "point_colour",
    "window_aspect", 
    "window_control", 
    "win_max_dim", 
    "zoom_magnification", 
    "zoom_mode", 
    "zoom_steps",
    ]

# def _image_viewer_settings(function):   
    
#     new_kwargs = {"default_image_viewer_settings" : default_image_viewer_settings}
    
#     def inner_function(**kwargs):
#         kwargs = {**new_kwargs, **kwargs}
#         return function(**kwargs)
#     return inner_function

# default_image_viewer_settings={
#     'window_aspect': 'normal', 
#     'window_control': 'internal', 
#     'window_max_dimension': 1000, 
#     'zoom_magnification': 0.5, 
#     'zoom_mode': 'continuous', 
#     'zoom_steps': 20
# }


#%% annotation functions

_annotation_functions = {
    "create_mask": "mask",
    "create_reference": "mask",
    "detect_mask": "mask",
    "detect_reference": "reference",
    "detect_contours": "contour",
    "edit_contours": "drawing",
    "enter_data": "comment",
    "set_landmarks": "landmark",
    }

_annotation_function_dicts = {
    "mask": {},
    "contour": {},
    "comment": {},
    "drawing": {},
    "landmark": {},
    "reference": {},
    }

#%% python helper functions

# from dataclasses import make_dataclass, dataclass

# def set_flags(fields):
    
#     @dataclass
#     class Flags:
    
#         def __post_init__(self):
#             [setattr(self, k, v) for k, v in self.fields.items()]
            
#     my_flags = Flags(fields)
                   
#     return my_flags    
    
# # @dataclass
# # class MyDataclass:
# #     data1: Optional[str] = None
# #     data2: Optional[Dict] = None
# #     data3: Optional[Dict] = None

# #     kwargs: field(default_factory=dict) = None

    
# flags = set_flags({"test": True})

# # flags.test

# # self.flags.skip = skip
# # self.flags.feedback = feedback
# # self.flags.terminate = False

# # def flags(flag_args):
    
    


# flags = dataclass_from_dict("flags", {"skip": False, "feedback": True, "terminate": False})
# flags.skip




#%% dependencies

pretty = PrettyPrinter(width=30)
