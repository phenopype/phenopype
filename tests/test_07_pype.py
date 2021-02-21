#%% modules
import os
import pytest

import phenopype as pp

from .settings import pype_name, wait_time


#%% tests

def test_pype(project_directory):
    test_params = {"flag_test_mode": True,
                   "wait_time": wait_time}
    p1 = pp.pype(project_directory, name=pype_name, test_params=test_params)
