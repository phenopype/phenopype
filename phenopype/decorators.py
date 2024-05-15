# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:02:39 2024

@author: mluerig
"""
import io
import warnings
from contextlib import redirect_stdout
from functools import wraps

from phenopype import utils_lowlevel as ul

#%% decorators 

def annotation_function(fun, *args, **kwargs):
    
    @wraps(fun)
    def annotation_function_wrapper(*args, **kwargs):
                
        ## determine the annotation type from function name
        kwargs["annotation_type"] = ul._get_annotation_type(fun.__name__)

        ## get annotation using 
        if "annotations" in kwargs:
            if kwargs["annotations"].__class__.__name__ in ["dict"]:
                if len(kwargs["annotations"]) > 0:                      
                    kwargs["annotation_id"] = ul._get_annotation_id(**kwargs)   
                    kwargs["annotation"] = ul._get_annotation2(**kwargs)
                else:
                    print("empty annotations supplied")
            else:
                print("wrong annotations data supplied - need dict")
                
        ## run function
        kwargs["annotation"] = fun(*args, **kwargs)
    
        ## return and update annotations    
        if "annotations" in kwargs:
            return ul._update_annotations(**kwargs)
        else:
            return kwargs["annotation"]
    
    ## close function wrapper
    return annotation_function_wrapper


def capture_stdout_log(fun, logger, *logger_args):
    
    @wraps(fun)
    def capture_stdout_log_wrapper():
                
        string_buffer = io.StringIO()
        with redirect_stdout(string_buffer):
                
            fun()
    
        ## reformat stdout
        stdout = string_buffer.getvalue()
        stdout_list = stdout.split("\n")
        for line in stdout_list:
            if not line == "":
                if line.endswith("\n"):
                    line = line[:-2]
                logger(logger_args)
                
    ## close function wrapper
    return capture_stdout_log_wrapper

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
    