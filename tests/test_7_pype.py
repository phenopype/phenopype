#%% modules
import os
import pytest

import phenopype as pp

from settings import pype_name


#%% tests

def test_pype(project_directory):
    p1 = pp.pype(project_directory, name=pype_name)


