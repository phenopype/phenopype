#%% modules
import mock
import os
import pytest
import phenopype as pp

from .settings import image_dir, stickle_image, pype_name, preset


#%% tests

@pytest.fixture(scope="module")
def directory_path():
    path = r"_temp/tests/project2/data/0__stickle1"
    directory_path = path
    return path

def test_load_directory(directory_path):
    directory = pp.utils.load_directory(directory_path)
    assert directory.__class__.__name__ == "container"

@pytest.fixture(scope="module")
def container(directory_path):
    print(os.listdir(directory_path))
    ct = pp.utils.load_directory(directory_path, save_suffix=pype_name)
    container = ct
    return ct

## actually tests load contours option of container
def test_container_load(container):
    container.load(contours=True)
    assert hasattr(container, "df_contours")
    
def test_load_pype_config(container):
    os.remove(os.path.join(container.dirpath, "pype_config_" + pype_name + ".yaml"))
    with mock.patch('builtins.input', return_value='y'):
        cfg = pp.utils_lowlevel._load_pype_config(container, 
                                                  preset=preset, 
                                                  config_name=pype_name)
    assert os.path.isfile(os.path.join(container.dirpath, "pype_config_" + pype_name + ".yaml"))
