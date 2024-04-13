#%% imports

clean_namespace = dir()
import_list = []

import importlib

## dlib
if not any([
        importlib.util.find_spec("phenomorph") is None,
        importlib.util.find_spec("dlib") is None,
        ]):
        
    import phenomorph, dlib
    
    import_list.append("phenomorph")
    clean_namespace.extend(["phenomorph", "dlib"])

## keras
if not any([
        importlib.util.find_spec("keras") is None,
        importlib.util.find_spec("tensorflow") is None,
        ]):
    
    import keras 
    
    import_list.extend(["keras"])
    clean_namespace.extend(["keras"])
    
## pytorch
if not any([
         importlib.util.find_spec("torch") is None,
         importlib.util.find_spec("torchvision") is None,
         ]):
     
     import torch, torchvision 
     
     import_list.extend(["torch"])
     clean_namespace.extend(["torch"])
     
## fastsam
if not any([
         importlib.util.find_spec("fastsam") is None,
         importlib.util.find_spec("ultralytics") is None,
         importlib.util.find_spec("torch") is None,
         importlib.util.find_spec("torchvision") is None,
         ]):
     
     import fastsam 
     
     import_list.extend(["fastsam"])
     clean_namespace.extend(["fastsam"])
        
        

#%% feedback

def __dir__():
    return clean_namespace 