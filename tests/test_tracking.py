#%% test header
import os
import cv2
import random

os.chdir("E:\\git_repos\\phenopype")
from phenopype import tracking

import pytest
# %% testing



image_dir = ".\\tutorials\\images"
image_list = os.listdir(image_dir)
image_str = random.choice(image_list)
image_arr = cv2.imread(os.path.join(".\\tutorials\\images", random.choice(image_list)))
img_choice = [image_str, image_arr]
image = random.choice(img_choice)

# 'motion tracker',

@pytest.fixture(scope="module")
def motion_tracker_obj():
    return tracking.motion_tracker(".\\tutorials\\videos\\isopods_fish.mp4")
    
def test_video_output(motion_tracker_obj):
    motion_tracker_obj.video_output(video_format="DIVX", save_colour=True, save_video=True, resize=0.5)
    assert motion_tracker_obj.name == 'isopods_fish.mp4'

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
