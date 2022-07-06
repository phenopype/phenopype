#%% imports

clean_namespace = dir()
import_list = []

import importlib

## phenomorph
if not importlib.util.find_spec("phenomorph") is None:
    
    import phenomorph, dlib
    
    import_list.append("phenomorph")
    clean_namespace.extend(["phenomorph", "dlib"])

## keras
if not any([
        importlib.util.find_spec("keras") is None,
        importlib.util.find_spec("tensorflow") is None,
        ]):
    
    # import tensorflow
    import keras 
    
    import_list.extend(["keras"])
    clean_namespace.extend(["keras"])
    

#%% feedback

def __dir__():
    return clean_namespace 