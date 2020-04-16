#%% modules
import os
import pytest

import phenopype as pp

from settings import pype_name


#%% tests

# @pytest.fixture(scope="session")
# def image(project_container):
#     image_path = os.path.join(project_container.dirpath, "canvas_" + pype_name + ".jpg")
#     image = pp.load_image(image_path)
#     return image

# def test_load_image(image):
#     assert image.__class__.__name__ == "ndarray"

# def test_show_image(image):
#     iv = pp.show_image(image)
#     assert image.__class__.__name__ == "ndarray"

def test_pype(project_directory):
    p1 = pp.pype(project_directory, name=pype_name)


