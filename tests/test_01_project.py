#%% modules
import os
import mock
import pytest
import time
import phenopype as pp

from .settings import root_dir1, image_dir, pype_name, wait_time, image_ref_filepath, config_target, config_replacement
from .settings import template_ex1, template_test1, template_test2

#%% tests

@pytest.fixture(scope="module")
def new_project():
    with mock.patch('builtins.input', return_value="y"):
        project = pp.project(root_dir=root_dir1, overwrite=True)
    new_project = project
    return project

def test_project(new_project):
    with mock.patch('builtins.input', return_value='n'):
        project = pp.project(root_dir=root_dir1)
    with mock.patch('builtins.input', return_value='y'):
        project = pp.project(root_dir=root_dir1)
    assert os.path.isdir(project.root_dir)    
        
def test_project_add_files(new_project):
    new_project.add_files(image_dir=image_dir, 
                      mode="link", 
                      include="stickle")
    new_project.add_files(image_dir=image_dir, 
                      include="stickle",
                      resize_factor=0.5,
                      overwrite=True)
    new_project.add_files(image_dir=image_dir, 
                      mode="copy", 
                      include="stickle",
                      overwrite=True)
    assert len(new_project.dirpaths) > 0
    
def test_project_add_config(new_project):
    new_project.add_config(name=pype_name, template="wrong-template")
    new_project.add_config(name=pype_name, template=None)
    new_project.add_config(name=pype_name, template=template_ex1)
    test_params = {"flag_test_mode": True,
                   "wait_time": wait_time}
    new_project.add_config(name=pype_name, template="wrong-template", 
                           overwrite=True, interactive=True, test_params=test_params)
    new_project.add_config(name=pype_name, template=template_test2, 
                           overwrite=True, interactive=True, test_params=test_params)
    new_project.add_config(name=pype_name, template=template_test2, interactive_image=2,
                           overwrite=True, interactive=True, test_params=test_params)
    new_project.add_config(name=pype_name, template=template_test2, interactive_image=new_project.dirnames[2+1],
                           overwrite=True, interactive=True, test_params=test_params)
    new_project.add_config(name=pype_name, template=template_test1, 
                           overwrite=True)
    config = pp.utils_lowlevel._load_yaml(os.path.join(new_project.root_dir, 
                                                       new_project.dirpaths[0], 
                                                       "pype_config_" + pype_name + ".yaml"))
    assert config["config_info"]["template_name"] == os.path.basename(template_test1)

def test_project_add_reference(new_project):
    test_params = {"flag_test_mode": True,
          "flag_tool": "reference",
          "reference_coords": [(701, 741), 
                           (1053, 774)],
          "point_list": [[(316, 675), 
                          (1236, 675), 
                          (1236, 1549), 
                          (316, 1549), 
                          (316, 675)]],
          "rect_list": [[316, 675, 1236, 1549]],
          "entry": "10",
          "wait_time": 100}
    new_project.add_reference(name="ref1", reference_image=image_ref_filepath,test_params=test_params)
    new_project.add_reference(name="ref1", reference_image=None,test_params=test_params)
    new_project.add_reference(name="ref1", reference_image="0__stickle1",test_params=test_params)
    new_project.add_reference(name="ref1", reference_image=0, test_params=test_params, overwrite=True)
    new_project.add_reference(name="ref1", reference_image=0, template=True, 
                              test_params=test_params)
    new_project.add_reference(name="ref1", reference_image=0, template=True, 
                              test_params=test_params, overwrite=True)
    assert os.path.isfile(os.path.join(root_dir1, "reference_ref1.tif"))
    
def test_project_edit_config(new_project):
    config_old = pp.utils_lowlevel._load_yaml(os.path.join(new_project.dirpaths[0], 
                                                           "pype_config_" + pype_name + ".yaml"))
    with mock.patch('builtins.input', return_value="y"):
        new_project.edit_config(name="v1", target=config_target, replacement=config_replacement)
    config_new = pp.utils_lowlevel._load_yaml(os.path.join(new_project.dirpaths[0], 
                                                           "pype_config_" + pype_name + ".yaml"))
    assert not config_old == config_new
    
def test_collect_results(new_project):
    new_project.collect_results(name="", files=["copy"], overwrite=True)
    assert len(os.listdir(os.path.join(root_dir1, "results"))) == 5

def test_project_save(new_project):
    pp.project.save(new_project, overwrite=True)
    assert os.path.isfile(os.path.join(new_project.root_dir, "project.data"))
