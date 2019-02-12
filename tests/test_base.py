#%% test header
import os
import cv2
import random

os.chdir("E:\\git_repos\\phenopype")
from phenopype import base

import pytest
# %% testing



image_dir = ".\\tutorials\\images"
image_list = os.listdir(image_dir)
image_str = random.choice(image_list)
image_arr = cv2.imread(os.path.join(".\\tutorials\\images", random.choice(image_list)))
img_choice = [image_str, image_arr]
image = random.choice(img_choice)

# 'project_maker',

@pytest.fixture(scope="module")
def project():
    return base.project_maker(".\\tutorials\\images")
#    assert len(project.filenames)>0
    
def test_project_maker(project):
    assert len(project.filenames)>0

#def test_median_grayscale_finder(project):
#    project.project_grayscale_finder()
#    assert len(project.filenames)>0

#def test_project_save(project):
#    project.project_grayscale_finder()
#    assert len(project.filenames)>0
#
#def test_project_update_filelist(project):
#    project.project_update_filelist()
#    assert len(project.filenames)>0



# 'scale_maker',
    
def test_scale_maker(project):
    template = project.filepaths[0]
    scale = base.scale_maker(template, value=10, unit="mm", show=False, zoom=False)
    return scale.measured
    assert len(scale.measured)>0










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
