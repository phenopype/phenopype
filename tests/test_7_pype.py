#%% modules
import os
import pytest

import phenopype as pp

from .settings import pype_name, flag_feedback


#%% tests

def test_pype(project_directory):
    if flag_feedback:
        p1 = pp.pype(project_directory, name=pype_name)
    else:
        p1 = pp.pype(project_directory, name=pype_name, feedback=False)
        

