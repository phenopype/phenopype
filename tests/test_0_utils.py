#%% modules
import os
import pytest

import phenopype as pp

from .settings import image_dir, stickle_image


#%% tests

@pytest.fixture(scope="module")
def path():
    if not os.getcwd() == r"/home/travis/build/mluerig/phenopype":
        os.chdir(r"E:\git_repos\phenopype")
    image_path = "tutorials/images/isopods.jpg"
    return image_path

def test_load_image(path):
    image = pp.load_image(path)
    assert image.__class__.__name__ == "ndarray"

