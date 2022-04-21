#%% import plugin functions


# from .libraries import ml_morph
from . import measurement

clean_namespace = dir()

#%% import plugin libraries

import_list = []

import importlib.util

if not importlib.util.find_spec("phenomorph") is None:
    from .libraries import ml_morph
    import_list.append("ml_morph")
    clean_namespace.append("ml_morph")
    
if not importlib.util.find_spec("keras") is None:
    from .libraries import keras
    import_list.append("keras")
    clean_namespace.append("keras")
    

#%% feedback

def __dir__():
    return clean_namespace 

if len(import_list) > 0:
    print("phenopype imported dependencies to the following plugins:")
    print(*import_list, sep=', ')

