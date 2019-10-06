#%% test header
import os
import cv2
import random

import phenopype as pp

import pytest
# %% testing setup

image_dir = ".\\tutorials\\images"

test_proj = pp.project_maker()
test_proj.add_files(image_dir)
random_img = random.choice(test_proj.filepaths)


#%% base

@pytest.fixture()
def pm():
    return pp.project_maker()
    
def test_project_maker(pm):
    pm.add_files(image_dir, search_mode="recursive")
    assert len(pm.filenames)==8

@pytest.fixture()
def sm():
    return pp.scale_maker(image=test_proj.filepaths[2])
    
def test_scale_maker(sm):
    sm.measure_scale(zoom_factor=5)
    sm.make_scale_template(mode="rectangle", show=False)
    scale_mask, scale_current = sm.detect_scale(target_image=test_proj.filepaths[4], min_matches=10, show=True)    
    assert len(scale_mask)==3


#%% tracking
    
@pytest.fixture(scope="module")
def motion_tracker_obj():
    return pp.motion_tracker(".\\tutorials\\videos\\isopods_fish.mp4")
    
def test_video_output(motion_tracker_obj):
    motion_tracker_obj.video_output(video_format="DIVX", save_colour=True, save_video=True, resize=0.5)
    assert motion_tracker_obj.name == 'isopods_fish.mp4'




# 'avgit',
# 'decode_fourcc',
# 'exif_date',
# 'find_centroid',
# 'find_skeleton',
# 'gray_scale',
# 'label_finder',
# 'landmark_module',
# 'motion_tracker',
# 'object_finder',
# 'polygon_maker',
# 'print_function',

# 'show_img',
# 'tracking_method',
