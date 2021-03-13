#%% modules
import os
import mock
import pytest

import phenopype as pp

from .settings import root_dir1, image_dir, pype_name, preset


#%% tests

@pytest.fixture(scope="module")
def new_project():
    with mock.patch('builtins.input', return_value='y'):
        project = pp.project(root_dir=root_dir1, overwrite=True)
    new_project = project
    return project

def test_project(new_project):
    with mock.patch('builtins.input', return_value='y'):
        project = pp.project(root_dir=root_dir1)
    assert os.path.isdir(root_dir1)
    
def test_project_add_files(new_project):
    new_project.add_files(image_dir=image_dir, 
                      raw_mode="link", 
                      include="stickle")
    new_project.add_files(image_dir=image_dir, 
                      raw_mode="copy", 
                      include="stickle",
                      overwrite=True)
    assert len(new_project.filepaths) > 0
    
def test_project_add_config(new_project):
    new_project.add_config(name=pype_name, config_preset="test")
    new_project.add_config(name=pype_name, config_preset=None, overwrite=True)
    cfg_link = "_temp/tests/project1/data/0__stickle1/pype_config_v1.yaml"
    new_project.add_config(name=pype_name, config_preset=cfg_link, overwrite=True)
    new_project.add_config(name=pype_name, config_preset=preset, overwrite=True)
    config = pp.utils_lowlevel._load_yaml(os.path.join(new_project.root_dir, 
                                                       new_project.dirpaths[0], 
                                                       "pype_config_" + pype_name + ".yaml"))
    assert config["pype"]["preset"] == preset

def test_project_add_scale(new_project):
    test_params = {"flag_test_mode": True,
          "flag_tool": "scale",
          "scale_coords": [(701, 741), 
                           (1053, 774)],
          "point_list": [[(316, 675), 
                          (1236, 675), 
                          (1236, 1549), 
                          (316, 1549), 
                          (316, 675)]],
          "rect_list": [[316, 675, 1236, 1549]],
          "entry": "10",
          "wait_time": 100}
    new_project.add_scale(reference_image="0__stickle1", test_params=test_params)
    new_project.add_scale(reference_image=0, test_params=test_params, overwrite=True)
    assert os.path.isfile(os.path.join(root_dir1, "scale_template.jpg"))

def test_collect_results(new_project):
    new_project.collect_results(name="", files=["raw"], overwrite=True)
    assert len(os.listdir(os.path.join(root_dir1, "results"))) == 5

def test_project_save(new_project):
    pp.project.save(new_project, overwrite=True)
    assert os.path.isfile(os.path.join(new_project.root_dir, "project.data"))
