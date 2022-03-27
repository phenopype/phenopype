#%% imports 

from . import measurement

clean_namespace = dir()


#%% namespace

def __dir__():
    return clean_namespace

#%% plugin imports 

try:
    import phenomorph
except ModuleNotFoundError:
    pass
    
