#%% modules

import numpy as np
import pytest

import phenopype as pp

from .settings import flag_overwrite

#%% tests

def test_blur(project_container):
    pp.segmentation.blur(project_container, kernel_size=3)
    assert not (project_container.image_copy==project_container.image).all()

def test_threshold(project_container):
    pp.segmentation.threshold(project_container, method="adaptive", 
                              blocksize=199, constant=5, channel="red",
                              invert=True)
    unique, counts = np.unique(project_container.image, return_counts=True)
    assert len(unique) == 2

def test_morphology(project_container):
    pp.segmentation.morphology(project_container, operation="open", 
                                shape="ellipse", kernel_size=3, iterations=3)
    assert not (project_container.image_bin==project_container.image).all()

def test_draw(project_container):
    pp.segmentation.draw(project_container, overwrite=flag_overwrite)
    assert not (project_container.image_bin==project_container.image).all()

def test_find_contours(project_container):
    pp.segmentation.find_contours(project_container, min_area=250)
    assert not (project_container.image_bin==project_container.image).all()