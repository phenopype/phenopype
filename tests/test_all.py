#%% test header
import os
import cv2
import random

import phenopype as pp

import pytest
import mock
# %% testing setup

root_dir = "./_temp/tests/project"
image_dir = "./tutorials/images"
pype_name = "v1"
dir_num = 0
preset = "demo1"
# test_proj = pp.project_maker()
# test_proj.add_files(image_dir)
# random_img = random.choice(test_proj.filepaths)


#%% project


@pytest.fixture(scope="module")
def project():
    with mock.patch('builtins.input', return_value='y'):
        project = pp.project(root_dir=root_dir, overwrite=True)
    project.add_files(image_dir=image_dir)
    return project

def test_project_add_files(project):
    assert len(project.filepaths) == len(os.listdir(image_dir))
    
def test_project_add_config(project):
    project.add_config(name=pype_name, config_preset=preset)
    config = pp.utils_lowlevel._load_yaml(os.path.join(project.dirpaths[dir_num], "pype_config_" + pype_name + ".yaml"))
    assert config["pype"]["preset"] == preset

def test_load_image(project):
    image = pp.load_image(project.filepaths[0])

def test_load_directory(project):
    ct = pp.load_directory(project.dirpaths[0])

def test_segmentation_blur(project):
    ct = pp.load_directory(project.dirpaths[0])
    pp.segmentation.blur(ct)
    pp.segmentation.morphology(ct)
    pp.segmentation.threshold(ct)
    pp.segmentation.watershed(ct)
    pp.segmentation.find_contours(ct)
    
# @pytest.fixture()
# def scale():
#     return pp.scale(image=test_proj.filepaths[2])
    
# def test_scale_maker(scale):
#     scale.measure(zoom_factor=5)
#     scale.create_template(mode="rectangle", show=False)
#     scale_detected, scale_mask, scale_current  = scale.detect(target_image=test_proj.filepaths[4], show=True, equalize = True)    
#     assert len(scale_mask)==3   


# #%% tracking
    
# @pytest.fixture(scope="module")
# def motion_tracker_obj():
#     return pp.motion_tracker(".\\tutorials\\videos\\isopods_fish.mp4")
    
# def test_video_output(motion_tracker_obj):
#     motion_tracker_obj.video_output(video_format="DIVX", save_colour=True, save_video=True, resize=0.5)
#     assert motion_tracker_obj.name == 'isopods_fish.mp4'




#%%