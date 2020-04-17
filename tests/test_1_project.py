#%% modules
import os
import mock
import pytest

import phenopype as pp

from .settings import root_dir1, root_dir2, image_dir, pype_name, ref_image, preset #, flag_overwrite


#%% tests

@pytest.fixture(scope="module")
def new_project():
    with mock.patch('builtins.input', return_value='y'):
        project = pp.project(root_dir=root_dir1, overwrite=True)
    project.add_files(image_dir=image_dir, 
                      raw_mode="link", 
                      include="stickle")
    return project

def test_project_add_files(new_project):
    assert len(new_project.filepaths) == 5
    
def test_project_add_config(new_project):
    new_project.add_config(name=pype_name, config_preset=preset)
    config = pp.utils_lowlevel._load_yaml(os.path.join(new_project.root_dir, 
                                                       new_project.dirpaths[0], 
                                                       "pype_config_" + pype_name + ".yaml"))
    assert config["pype"]["preset"] == preset

def test_project_save(new_project):
    pp.project.save(new_project, overwrite=True)
    assert os.path.isfile(os.path.join(new_project.root_dir, "project.data"))
