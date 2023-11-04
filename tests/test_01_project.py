#%% modules
import os
import mock
import pytest

import phenopype as pp

#%% tests

def test_project(settings):
    with mock.patch('builtins.input', return_value='n'):
        project = pp.Project(root_dir=pytest.project_root_dir_1)
    with mock.patch('builtins.input', return_value='y'):
        project = pp.Project(root_dir=pytest.project_root_dir_1)
    with mock.patch('builtins.input', return_value='n'):
        project = pp.Project(root_dir=pytest.project_root_dir_1, overwrite=True)
    with mock.patch('builtins.input', return_value='y'):
        project = pp.Project(root_dir=pytest.project_root_dir_1, overwrite=True)
        
    project = pp.Project(root_dir=pytest.project_root_dir_1)
        
    assert os.path.isdir(project.root_dir)    
    
    
        
def test_project_add_files(project, settings):
    
    project.add_files(
        image_dir=pytest.image_dir, 
        mode="link", 
        include="stickle"
        )
    
    project.add_files(
        image_dir=pytest.image_dir, 
        include="stickle",
        resize_factor=0.5,
        overwrite=True
        )
    
    project.add_files(
        image_dir=pytest.image_dir, 
        mode="copy", 
        include="stickle",
        overwrite=True
        )
    
    assert len(project.dir_paths) > 0
    
    
    
def test_project_add_config(project, settings):
       
    project.add_config(
        tag=pytest.tag_1, 
        template_path="wrong-template"
        )
    
    project.add_config(
        tag=pytest.tag_1, 
        template_path=pytest.template_path_1
        )

    config = pp.utils_lowlevel._load_yaml(
        os.path.join(
            project.root_dir, 
            project.dir_paths[0], 
            "pype_config_" + pytest.tag_1 + ".yaml"
            ))
    
    assert config["config_info"]["template_name"] == os.path.basename(pytest.template_path_1)
    
    

def test_project_add_reference(project, reference_detected, mask_polygon, settings):
    
    annotations = {**reference_detected, **mask_polygon}
    
    project.add_reference(
        tag=pytest.tag_1, 
        reference_image_path=pytest.reference_image_path,
        reference_tag="ref1",
        feedback=False,
        overwrite=True,
        annotations=annotations,
        )
    
    assert os.path.isfile(os.path.join(project.root_dir, "reference", "ref1_search_template.tif"))
    
    
    
def test_project_edit_config(project, settings):
    
    with mock.patch('builtins.input', return_value="y"):
        project.edit_config(
            tag=pytest.tag_1, 
            target=pytest.edit_config_target, 
            replacement=pytest.edit_config_replacement
            )
        
    path = os.path.join(project.dir_paths[0],  "pype_config_" + pytest.tag_1 + ".yaml")
    
    with open(path) as f:
        if pytest.edit_config_replacement in f.read():
            success = True
        else:
            success = False
            
    assert success
    


