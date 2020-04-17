#%% modules

import os
import pytest

import phenopype as pp

from settings import root_dir2, image_dir, pype_name, preset, stickle_image

#%% project

@pytest.fixture(scope="session")
def project_container():
    if os.path.isfile(os.path.join(root_dir2, "project.data")):
        project = pp.project.load(root_dir2)
    else: 
        project = pp.project(root_dir=root_dir2, overwrite=False)
        project.add_files(image_dir=image_dir, 
                          raw_mode="link", 
                          include="stickle")
        project.add_config(name=pype_name, config_preset=preset)
        pp.project.save(project)
    print(os.getcwd())
    print(project.dirpaths)
    print(project.dirpaths[stickle_image])
    ct = pp.load_directory(project.dirpaths[stickle_image])
    ct.load(save_suffix=pype_name)
    return ct

@pytest.fixture(scope="session")
def project_directory():
    if os.path.isfile(os.path.join(root_dir2, "project.data")):
        project = pp.project.load(root_dir2)
    else: 
        project = pp.project(root_dir=root_dir2, overwrite=False)
        project.add_files(image_dir=image_dir, 
                          raw_mode="link", 
                          include="stickle")
        project.add_config(name=pype_name, config_preset=preset)
        pp.project.save(project)
    return project.dirpaths[stickle_image]
