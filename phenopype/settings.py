#%% load modules

import cv2
import os
from importlib.resources import path

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
pype_config_templates = {}
with path(__package__, 'templates') as template_dir:
    template_list = os.listdir(template_dir)
    for template_name in template_list:
        if template_name.endswith(".yaml"):
            template_path = os.path.join(template_dir, template_name)
            pype_config_templates[template_name] = template_path
            
            
            
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


#%% default arguments



_image_viewer_arg_list = [
    "window_aspect", 
    "window_control", 
    "window_max_dimension", 
    "zoom_magnification", 
    "zoom_mode", 
    "zoom_steps"]

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



