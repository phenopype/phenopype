#%% modules

import os
import pytest 
from shutil import copyfile

import phenopype as pp

from .settings import pype_name

#%% test

def test_select_canvas(project_container):
    pp.visualization.select_canvas(project_container, canvas="red", multi=True)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_draw_contours(project_container):
    pp.export.save_contours(project_container, overwrite=True)    
    pp.export.save_contours(project_container, overwrite=True, save_coords=True)    
    contour_source = os.path.join(project_container.dirpath, "contours_" + pype_name + ".csv")
    contour_dest = os.path.join(project_container.dirpath, "contours_" + "v2" + ".csv")
    
    copyfile(contour_source, contour_dest)
    pp.visualization.draw_contours(project_container, label=True, mark_holes=True,
                                   compare = "v2", fill=0.5, skeleton=True)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_draw_landmarks(project_container):
    pp.visualization.draw_landmarks(project_container)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_draw_masks(project_container):
    pp.visualization.draw_masks(project_container, label=True)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_draw_polylines(project_container):
    pp.visualization.draw_polylines(project_container)
    assert not (project_container.image_copy==project_container.canvas).all()
