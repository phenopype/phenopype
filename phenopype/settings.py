#%% load modules

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

flag_verbose = False