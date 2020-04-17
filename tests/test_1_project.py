#%% test header
import os
import cv2
import random

import phenopype as pp

import pytest
import mock
# %% testing setup

root_dir1 = "tests/resources/project1"
root_dir2 = "tests/resources/project2"
image_dir = "tutorials/images"
pype_name = "v1"
ref_image = "0__stickleback_side"
preset = "demo1"

flag_overwrite_project = False
# test_proj = pp.project_maker()
# test_proj.add_files(image_dir)
# random_img = random.choice(test_proj.filepaths)


#%% project

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
        with mock.patch('builtins.input', return_value='y'):
            project = pp.project(root_dir=root_dir2, overwrite=True)
        project.add_files(image_dir=image_dir, 
                          raw_mode="link", 
                          include="stickle")
        project.add_config(name=pype_name, config_preset=preset)
        pp.project.save(project)
    return project

def test_project_add_scale(saved_project):
    saved_project.add_scale(reference_image=ref_image, template=True)    
