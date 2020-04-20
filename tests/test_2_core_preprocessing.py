#%% modules

import os
import pytest

import phenopype as pp

from .settings import flag_overwrite

#%% tests

def test_create_mask(project_container):
    pp.preprocessing.create_mask(project_container, tool="polygon", overwrite=flag_overwrite)
    assert len(project_container.df_masks) > 0

def test_find_scale(project_container):
    pp.preprocessing.find_scale(project_container, overwrite=True)
    assert len(project_container.df_masks) > 0

def test_create_scale(project_container):
    pp.preprocessing.create_scale(project_container, mask=True, overwrite=False)
    assert len(project_container.df_masks) > 0

def test_enter_data(project_container):
    pp.preprocessing.enter_data(project_container, columns="ID", overwrite=flag_overwrite)
    assert len(project_container.df_masks) > 0

def test_invert_image(project_container):
    pp.preprocessing.invert_image(project_container)
    assert len(project_container.df_masks) > 0

