#%% modules
import mock
import os
import pytest
import phenopype as pp
import shutil

from .settings import image_dir, stickle_image, image_path, image_save_dir


#%% tests

@pytest.fixture(scope="module")
def path():
    path = image_path
    return image_path

def test_load_image_arr(path):
    image, df = pp.load_image(path, df=True)
    assert all([image.__class__.__name__ == "ndarray",
                df["filename"][0] == "isopods.jpg"])

def test_load_image_ct(path):
    image = pp.load_image(path, meta=True, cont=True, df=True)
    assert image.__class__.__name__ == "container"

def test_load_meta_data(path):
    meta = pp.utils.load_meta_data(path, show_fields=True)
    assert meta["ISOSpeedRatings"] == "200"

@pytest.fixture(scope="module")
def image(path):
    if os.path.isdir(image_save_dir):
        shutil.rmtree(image_save_dir) 
    with mock.patch('builtins.input', return_value='y'):
        img = pp.load_image(path, dirpath=image_save_dir)
    image = img
    return img

def test_show_image(image):
    test_params = {"flag_test_mode": True,
                   "wait_time": 1000}
    pp.show_image(image, test_params=test_params)
    test_params = {"flag_test_mode": True,
                   "wait_time": 10}
    img_list = []
    for i in range(11):
        img_list.append(image)
    with mock.patch('builtins.input', return_value='y'):
        pp.show_image(img_list, 
                      test_params=test_params, 
                      reset_position=True)
        
def test_save_image(image):
    pp.save_image(image, name="test_img", dirpath=image_save_dir)
