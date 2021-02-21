#%% modules
import mock
import os
import time
import phenopype as pp

from .settings import pype_name


#%% tests

def test_save_canvas(project_container):
    pp.export.save_canvas(project_container.canvas)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_canvas(project_container, dirpath="new")
    pp.export.save_canvas(project_container, overwrite=True)
    pp.export.save_canvas(project_container, overwrite=False)
    assert os.path.isfile(os.path.join(project_container.dirpath, "canvas_" + pype_name + ".jpg"))

def test_save_landmarks(project_container):
    pp.export.save_landmarks(project_container.df_landmarks)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_landmarks(project_container, dirpath="new")        
    pp.export.save_landmarks(project_container, overwrite=True)
    pp.export.save_landmarks(project_container, overwrite=False)
    assert os.path.isfile(os.path.join(project_container.dirpath, "landmarks_" + pype_name + ".csv"))
    
def test_save_colours(project_container):
    pp.export.save_colours(project_container.df_colours)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_colours(project_container, dirpath="new")        
    pp.export.save_colours(project_container, overwrite=True)
    pp.export.save_colours(project_container, overwrite=False)    
    assert os.path.isfile(os.path.join(project_container.dirpath, "colours_" + pype_name + ".csv"))

def test_save_contours(project_container):
    pp.export.save_contours(project_container.df_colours)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_contours(project_container, dirpath="new")        
    pp.export.save_contours(project_container, overwrite=True, save_coords=True)
    pp.export.save_contours(project_container, overwrite=False)    
    assert os.path.isfile(os.path.join(project_container.dirpath, "contours_" + pype_name + ".csv"))

def test_save_masks(project_container):
    pp.export.save_masks(project_container.df_masks)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_masks(project_container, dirpath="new")        
    pp.export.save_masks(project_container, overwrite=True)
    pp.export.save_masks(project_container, overwrite=False)    
    assert os.path.isfile(os.path.join(project_container.dirpath, "masks_" + pype_name + ".csv"))
    
def test_save_polylines(project_container):
    pp.export.save_polylines(project_container.df_polylines)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_polylines(project_container, dirpath="new")        
    pp.export.save_polylines(project_container, overwrite=True)
    pp.export.save_polylines(project_container, overwrite=False)    
    assert os.path.isfile(os.path.join(project_container.dirpath, "polylines_" + pype_name + ".csv"))

def test_save_drawings(project_container):
    pp.export.save_drawings(project_container.df_drawings)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_drawings(project_container, dirpath="new")        
    pp.export.save_drawings(project_container, overwrite=True)
    pp.export.save_drawings(project_container, overwrite=False)    
    assert os.path.isfile(os.path.join(project_container.dirpath, "drawings_" + pype_name + ".csv"))
    
    
def test_save_data_entry(project_container):
    pp.export.save_data_entry(project_container.df_other_data)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_data_entry(project_container, dirpath="new")        
    pp.export.save_data_entry(project_container, overwrite=True)
    pp.export.save_data_entry(project_container, overwrite=False)    
    attr = pp.utils_lowlevel._load_yaml(os.path.join(project_container.dirpath, "attributes.yaml"))
    assert "other" in attr    
    
def test_save_reference(project_container):
    pp.export.save_reference(project_container.reference_detected_px_mm_ratio)
    with mock.patch('builtins.input', return_value='n'):
        pp.export.save_reference(project_container, dirpath="new")        
    pp.export.save_reference(project_container, overwrite=True)
    pp.export.save_reference(project_container, overwrite=False)  
    attr = pp.utils_lowlevel._load_yaml(os.path.join(project_container.dirpath, "attributes.yaml"))
    assert "reference" in attr
