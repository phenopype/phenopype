# -*- coding: utf-8 -*-

__all__ = []

import pkgutil
import inspect

<<<<<<< HEAD
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
=======
#__path__="E:\\Python1\\phenopype\\"
>>>>>>> parent of 1439bed... versioneer added

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)

    for name, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue

        globals()[name] = value
        __all__.append(name)
<<<<<<< HEAD
        
=======
        
#from phenopype import utils, base, custom
#pp.__all__ + ["utils", "base", "custom"]
>>>>>>> parent of 1439bed... versioneer added
