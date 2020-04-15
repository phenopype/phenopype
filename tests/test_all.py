#%% test header
import os
import cv2
import random

import phenopype as pp

import pytest
# %% testing setup

image_dir = ".\\tutorials\\images"

# test_proj = pp.project_maker()
# test_proj.add_files(image_dir)
# random_img = random.choice(test_proj.filepaths)


#%% base

@pytest.fixture()
def pm():
    return pp.project(root_dir="./tutorials")
    
def test_project_maker(pm):
    pm.add_files(image_dir, search_mode="recursive")
    assert len(pm.filenames)>3

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




