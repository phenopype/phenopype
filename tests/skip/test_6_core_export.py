#%% modules
import os
import time

import phenopype as pp

from .settings import pype_name


#%% tests

def test_save_canvas(project_container):
    pp.export.save_canvas(project_container, overwrite=True)
    time_diff = time.time() - os.path.getmtime(os.path.join(project_container.dirpath, "canvas_" + pype_name + ".jpg"))
    assert time_diff < 2

def test_save_landmarks(project_container):
    pp.export.save_landmarks(project_container, overwrite=True)
    time_diff = time.time() - os.path.getmtime(os.path.join(project_container.dirpath, "landmarks_" + pype_name + ".csv"))
    assert time_diff < 2
    
def test_save_colours(project_container):
    pp.export.save_colours(project_container, overwrite=True)
    time_diff = time.time() - os.path.getmtime(os.path.join(project_container.dirpath, "colours_" + pype_name + ".csv"))
    assert time_diff < 2
    
def test_save_contours(project_container):
    pp.export.save_contours(project_container, overwrite=True)
    time_diff = time.time() - os.path.getmtime(os.path.join(project_container.dirpath, "contours_" + pype_name + ".csv"))
    assert time_diff < 2
    
def test_save_drawing(project_container):
    pp.export.save_drawing(project_container, overwrite=True)
    attr = pp.utils_lowlevel._load_yaml(os.path.join(project_container.dirpath, "attributes.yaml"))
    assert "drawing" in attr
    
def test_save_data_entry(project_container):
    pp.export.save_data_entry(project_container, overwrite=True)
    attr = pp.utils_lowlevel._load_yaml(os.path.join(project_container.dirpath, "attributes.yaml"))
    assert "other" in attr

def test_save_masks(project_container):
    pp.export.save_masks(project_container, overwrite=True)
    time_diff = time.time() - os.path.getmtime(os.path.join(project_container.dirpath, "masks_" + pype_name + ".csv"))
    assert time_diff < 2
    
def test_save_polylines(project_container):
    pp.export.save_polylines(project_container, overwrite=True)
    time_diff = time.time() - os.path.getmtime(os.path.join(project_container.dirpath, "polylines_" + pype_name + ".csv"))
    assert time_diff < 2
    
def test_save_scale(project_container):
    pp.export.save_scale(project_container, overwrite=True)
    attr = pp.utils_lowlevel._load_yaml(os.path.join(project_container.dirpath, "attributes.yaml"))
    assert "scale" in attr
