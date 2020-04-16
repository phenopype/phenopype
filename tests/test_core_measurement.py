#%% test header
import os
import cv2
import random
import time

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
    ct = pp.load_directory(project.dirpaths[0])
    ct.load(save_suffix=pype_name)
    return ct
    
def test_landmarks(project_container):
    pp.measurement.landmarks(project_container, point_size=10)
    assert project_container.df_landmarks.__class__.__name__ == "DataFrame"

def test_draw_landmarks(project_container):
    pp.visualization.draw_landmarks(project_container, point_size=10)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_save_landmarks(project_container):
    pp.export.save_landmarks(project_container, overwrite=True, save_suffix=pype_name)
    time_diff = time.time()-os.path.getmtime(os.path.join(project_container.dirpath, "landmarks_" + pype_name + ".csv"))
    assert time_diff < 2