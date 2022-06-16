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


def test_template(settings, project):
    
    pp.load_template(
        "string",
        dir_path=pytest.test_dir,
        )
    
    pp.load_template(
        2,
        dir_path=pytest.test_dir,
        )
    
    pp.load_template(
        pytest.template_path_1,
        )
    
    pp.load_template(
        pytest.template_path_1,
        image_path=pytest.image_path,
        )
    
    pp.load_template(
        pytest.template_path_1,
        image_path=pytest.image_path,
        )
    
    pp.load_template(
        pytest.template_path_1,
        image_path=pytest.image_path,
        overwrite=True,
        keep_comments = False,
        )
    
    pp.load_template(
        pytest.template_path_1,
        dir_path=pytest.test_dir,
        )

    assert os.path.isfile(os.path.join(pytest.test_dir, "pype_config_v1.yaml"))



def test_show_image(image):

    pp.show_image(image, feedback=False)

    with mock.patch('builtins.input', return_value="y"):
        pp.show_image([image, image, image, image, image, 
                       image, image, image, image, image, 
                       image],
                      feedback=False)
    
    
def test_save_image(image):

    pp.save_image(image, file_name="test", dir_path=pytest.test_dir)
    pp.save_image(image, file_name="test", dir_path=pytest.test_dir)
    pp.save_image(image, file_name="test", dir_path=pytest.test_dir, overwrite=True)


    assert os.path.isfile(os.path.join(pytest.test_dir, "test.jpg"))
