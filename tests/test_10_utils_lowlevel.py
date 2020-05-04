#%% modules
import mock
import os
import pytest
import phenopype as pp

from .settings import pype_name, preset


#%% tests

## actually tests load contours option of container
def test_container_load(project_container):
    container.load(contours=True)
    assert hasattr(container, "df_contours")
    
def test_load_pype_config(project_container):
    os.remove(os.path.join(container.dirpath, "pype_config_" + pype_name + ".yaml"))
    with mock.patch('builtins.input', return_value='y'):
        cfg = pp.utils_lowlevel._load_pype_config(container, 
                                                  preset=preset, 
                                                  config_name=pype_name)
    assert os.path.isfile(os.path.join(container.dirpath, "pype_config_" + pype_name + ".yaml"))
