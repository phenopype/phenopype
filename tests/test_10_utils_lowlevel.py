#%% modules
import mock
import os
import pytest
import phenopype as pp

from .settings import pype_name, template_test1


#%% tests

## actually tests load contours option of container
def test_container_load(project_container):
    delattr(project_container, "df_contours")
    project_container.load(contours=True, save_suffix=pype_name)
    assert hasattr(project_container, "df_contours")
    
# def test_load_pype_config(project_container):
#     os.remove(os.path.join(project_container.dirpath, "pype_config_" + pype_name + ".yaml"))
#     with mock.patch('builtins.input', return_value='y'):
#         cfg = pp.utils_lowlevel._load_pype_config(project_container, 
#                                                   template=template_name, 
#                                                   config_name=pype_name)
#     assert os.path.isfile(os.path.join(project_container.dirpath, "pype_config_" + pype_name + ".yaml"))
