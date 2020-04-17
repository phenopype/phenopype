#%% modules
import os
import cv2
import random

import phenopype as pp

from settings import root_dir1, root_dir2, image_dir, pype_name, ref_image, preset, flag_overwrite

import pytest
import mock


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
    config = pp.utils_lowlevel._load_yaml(os.path.join(new_project.dirpaths[0], "pype_config_" + pype_name + ".yaml"))
    assert config["pype"]["preset"] == preset

def test_project_save(new_project):
    pp.project.save(new_project)
    assert os.path.isfile(os.path.join(new_project.root_dir, "project.data"))
    
    
@pytest.fixture(scope="module")
def saved_project():
    if os.path.isfile(os.path.join(root_dir2, "project.data")):
        project = pp.project.load(root_dir2)
    else: 
        project = pp.project(root_dir=root_dir2, overwrite=False)
        project.add_files(image_dir=image_dir, 
                          raw_mode="link", 
                          include="stickle")
        project.add_config(name=pype_name, config_preset=preset)
        pp.project.save(project)
    return project
    
    
def test_project_add_scale(saved_project):
    saved_project.add_scale(reference_image=ref_image, template=True)    
