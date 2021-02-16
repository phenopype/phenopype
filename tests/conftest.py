import os
import mock
import pytest
import shutil    
import phenopype as pp

from .settings import root_dir2, video_path, image_dir, image_save_dir, image_filepath, pype_name, template_test1, flag_overwrite

#%% project


@pytest.fixture(scope="session")
def project_container():
    with mock.patch('builtins.input', return_value='y'):
        proj = pp.project(root_dir=root_dir2, overwrite=flag_overwrite)
    proj.add_files(image_dir=image_dir, 
                      raw_mode="copy", 
                      include="stickle")
    proj.add_config(name=pype_name, template=template_test1)
    pp.project.save(proj)
    proj = pp.project.load(root_dir2)
    obj_input = pp.load_directory(proj.dirpaths[0], save_suffix=pype_name)
    project_container = obj_input
    return obj_input 

@pytest.fixture(scope="session")
def project_directory():
    proj = pp.project.load(root_dir2)
    image = proj.dirpaths[0]
    # project_directory = image 
    return image

@pytest.fixture(scope="session")
def motion_tracker():
    mt = pp.motion_tracker(video_path)
    motion_tracker = mt
    return mt

@pytest.fixture(scope="session")
def tracking_method():
    fish = pp.tracking_method(label="fish", remove_shadows=True, min_length=30,
                              overlay_colour="red", mode="single",
                              blur=15, # bigger blurring kernel
                              threshold=200 #higher sensitivity
                             )
    isopod = pp.tracking_method(label="isopod", remove_shadows=True, max_length=30,
                                overlay_colour="green", mode="multiple",
                                blur=9, # smaller blurring kernel
                                threshold=180, # lower sensitivity
                                operations=["diameter",  # isopod size
                                            "area",      # isopod area
                                            "grayscale", # isopod pigmentation
                                            "grayscale_background"] # background darkness
                               )
    methods = [fish, isopod]
    tracking_method = methods
    return methods

@pytest.fixture(scope="session")
def image_path():
    return image_filepath

@pytest.fixture(scope="session")
def image_array(image_path):
    if os.path.isdir(image_save_dir):
        shutil.rmtree(image_save_dir) 
    with mock.patch('builtins.input', return_value='y'):
        img = pp.load_image(image_path, dirpath=image_save_dir)
    image = img
    return img