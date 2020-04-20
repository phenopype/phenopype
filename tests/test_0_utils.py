#%% modules
import os
import pytest

import phenopype as pp

from .settings import image_dir, stickle_image


#%% tests

@pytest.fixture(scope="module")
def path():
    image_path = os.path.join(image_dir, os.listdir(image_dir)[stickle_image])
    return image_path

def test_load_image(path):
    image = pp.load_image(path)
    assert image.__class__.__name__ == "ndarray"

