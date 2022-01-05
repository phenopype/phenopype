#%% modules
import os
import mock
import pytest
import shutil 

import phenopype as pp


#%% tests

@pytest.fixture()
def refresh_test_dir(settings):
    if os.path.isdir(pytest.test_dir):
       shutil.rmtree(pytest.test_dir) 
       print("Removed existing test dir {}".format(os.path.abspath(pytest.test_dir)))
    os.makedirs(pytest.test_dir)

def test_prep(refresh_test_dir):
    pass