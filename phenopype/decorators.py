# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:02:39 2024

@author: mluerig
"""
import io
import os
import warnings
from contextlib import redirect_stdout
from functools import wraps

from phenopype import _vars
from phenopype import utils_lowlevel as ul

#%% decorators 

def annotation_function(fun):
    @wraps(fun)
    def annotation_function_wrapper(*args, **kwargs):
        # Determine the annotation type from function name
        kwargs["fun_name"] = fun.__name__
        kwargs["annotation_type"] = ul._get_annotation_type(fun.__name__)
        
        # Get annotation if provided
        if "annotations" in kwargs:
            if isinstance(kwargs["annotations"], dict):
                if len(kwargs["annotations"]) > 0:
                    kwargs["annotation_id"] = ul._get_annotation_id(**kwargs)
                    kwargs["annotation"] = ul._get_annotation2(**kwargs)

        ## Optional GUI-related logic
        kwargs["gui_data"] = {_vars._coord_type: ul._get_GUI_data(kwargs.get("annotation"))}
        kwargs["gui_settings"] = ul._get_GUI_settings(kwargs, kwargs.get("annotation"))

        # Run the wrapped function
        kwargs["annotation"] = fun(*args, **kwargs)

        # Update annotations and return results
        if "annotations" in kwargs:
            result = ul._update_annotations(**kwargs)
        else:
            result = kwargs["annotation"]

        return result

    return annotation_function_wrapper

# def capture_stdout_log(fun, logger, *logger_args):
    
#     @wraps(fun)
#     def capture_stdout_log_wrapper():
                
#         string_buffer = io.StringIO()
#         with redirect_stdout(string_buffer):  
#             fun()
    
#         ## reformat stdout
#         stdout = string_buffer.getvalue()
#         stdout_list = stdout.split("\n")
#         for line in stdout_list:
#             if not line == "":
#                 if line.endswith("\n"):
#                     line = line[:-2]
#                 logger(logger_args)
                
#     ## close function wrapper
#     return capture_stdout_log_wrapper

# def capture_print(func):
#     def wrapper(*args, **kwargs):
#         # Redirect stdout to capture print output
#         old_stdout = sys.stdout
#         sys.stdout = captured_output = StringIO()

#         try:
#             result = func(*args, **kwargs)
#             output = captured_output.getvalue()
#         finally:
#             # Reset stdout
#             sys.stdout = old_stdout
        
#         return result, output

#     return wrapper

def deprecation_warning(new_func=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn(f"{func.__name__} is deprecated; use {new_func.__name__} instead.", category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return new_func(*args, **kwargs)
        return wrapper
    return decorator
    

def legacy_args(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        if fun.__name__ == "save_image":
            if 'file_path' not in kwargs or not kwargs['file_path']:
                file_name = kwargs.get('file_name')
                dir_path = kwargs.get('dir_path')
                if file_name and dir_path:
                    kwargs['file_path'] = os.path.join(dir_path, file_name)
                    ul._print(f"WARNING - using 'file_name' and 'dir_path' is deprecated for >{fun.__name__}< - use 'file_path' instead", lvl=1)
        
        elif fun.__name__ == "save_canvas":
            if 'file_path' not in kwargs or not kwargs['file_path']:
                file_name = kwargs.get('file_name')
                dir_path = kwargs.get('dir_path')
                if file_name and dir_path:
                    kwargs['file_path'] = os.path.join(dir_path, file_name)
                    print(f"Warning - using file_name and dir_path is deprecated for {fun.__name__} - use file_path instead")
        
        return fun(*args, **kwargs)
    
    return wrapper