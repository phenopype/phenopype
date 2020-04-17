#%% modules
import sys
import phenopype as pp
import pytest

from settings import *


#%% tests

def test_landmarks(project_container):
    pp.measurement.landmarks(project_container, point_size=10, overwrite=flag_overwrite)
    assert project_container.df_landmarks.__class__.__name__ == "DataFrame"

def test_colour_intensity(project_container):
    pp.measurement.colour_intensity(project_container, channels=["gray","rgb"])
    assert project_container.df_colours.__class__.__name__ == "DataFrame"

def test_polylines(project_container):
    pp.measurement.polylines(project_container, line_width=5, overwrite=flag_overwrite)
    assert project_container.df_polylines.__class__.__name__ == "DataFrame"

@pytest.mark.skipif(os.getcwd() == "/home/travis/build/mluerig/phenopype")
def test_skeletonize(project_container):
    pp.measurement.skeletonize(project_container)
    assert "skeleton_coords" in project_container.df_contours