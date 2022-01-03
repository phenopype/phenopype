#%% modules

import mock
import pytest
import os

import phenopype as pp

#%% tests


def test_load_yaml():

    y = pp.utils_lowlevel._load_yaml(pytest.template_path_4, legacy=True, typ="safe")
    y = pp.utils_lowlevel._load_yaml(pytest.template_path_1)

    pp.utils_lowlevel._show_yaml(y)

