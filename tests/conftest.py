#%% modules

import os
import mock
import pytest
    
import phenopype as pp

from settings import root_dir2, image_dir, pype_name, preset, ref_image, stickle_image, flag_overwrite

#%% project

@pytest.fixture(scope="session")
def project_container():
    if os.path.isfile(os.path.join(root_dir2, "project.data")):
        project = pp.project.load(root_dir2)
    else: 
        with mock.patch('builtins.input', return_value='y'):
            project = pp.project(root_dir=root_dir2, overwrite=flag_overwrite)
        project.add_files(image_dir=image_dir, 
                          raw_mode="link", 
                          include="stickle")
        project.add_config(name=pype_name, config_preset=preset)
        project.add_scale(reference_image=ref_image, template=True)    
        pp.project.save(project)
    ct = pp.load_directory(os.path.join(project.root_dir, 
                                        project.dirpaths[stickle_image]))
    ct.load(save_suffix=pype_name)
    return ct

@pytest.fixture(scope="session")
def project_directory():
    if os.path.isfile(os.path.join(root_dir2, "project.data")):
        project = pp.project.load(root_dir2)
    else: 
        with mock.patch('builtins.input', return_value='y'):
            project = pp.project(root_dir=root_dir2, overwrite=flag_overwrite)
        project.add_files(image_dir=image_dir, 
                          raw_mode="link", 
                          include="stickle")
        project.add_config(name=pype_name, config_preset=preset)
        project.add_scale(reference_image=ref_image, template=True)    
        pp.project.save(project)
    directory = os.path.join(project.root_dir, 
                             project.dirpaths[stickle_image])
    return directory