#%% modules

import mock
import pytest
import os

import phenopype as pp



#%% tests



def test_load_image(settings, project):
    
    project.add_files(
        image_dir=pytest.image_dir, 
        mode="link", 
        include="stickle"
        )
    
    image = pp.load_image(pytest.template_path_1)
    image = pp.load_image(project.dir_paths[0])    
    image = pp.load_image(project.root_dir)    
    image = pp.load_image("string")
    image = pp.load_image(2)
    
    image = pp.load_image(pytest.image_path, mode="colour")
    image = pp.load_image(pytest.image_path, mode="gray")
    image = pp.load_image(pytest.image_path)

    assert image.__class__.__name__ == "ndarray"

   
def test_save_image(image):

    pp.save_image(image, file_name="test.jpg", dir_path=pytest.test_dir)
    pp.save_image(image, file_name="test.jpg", dir_path=pytest.test_dir, overwrite=True)


    assert os.path.isfile(os.path.join(pytest.test_dir, "test.jpg"))
