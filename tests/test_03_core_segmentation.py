#%% modules

import copy
import numpy as np
import pytest 

import phenopype as pp

#%% fixtures


@pytest.fixture
def drawing():

    return {"drawing": {'a': {'info': {'annotation_type': 'drawing',
   'phenopype_function': 'edit_contour',
   'phenopype_version': '3.0.dev0'},
  'settings': {'overlay_blend': 0.2,
   'overlay_line_width': 1,
   'overlay_colour_left': (0, 128, 0),
   'overlay_colour_right': (0, 0, 255)},
  'data': {'drawing': [[[(1375, 328), (1375, 328)], 0, 16],
    [[(1416, 336), (1416, 336)], 0, 16],
    [[(1445, 346), (1445, 346)], 0, 16],
    [[(1474, 350), (1474, 350)], 0, 16],
    [[(1677, 390), (1677, 390)], 0, 16],
    [[(1708, 395), (1708, 395)], 0, 16]]}}}}


#%% tests

def test_threshold(image, reference_detected, mask_polygon):
    
    annotations = copy.deepcopy({**reference_detected, **mask_polygon})

    thresh = pp.segmentation.threshold(
        image, 
        method="adaptive", 
        blocksize=100, 
        constant=10, 
        )
    
    thresh = pp.segmentation.threshold(
        image, 
        method="binary",
        annotations=annotations,
        )
    
    annotations["mask"]["a"]["data"]["include"] = False
    
    thresh = pp.segmentation.threshold(
        image, 
        method="binary",
        annotations=annotations,
        )
    
    thresh = pp.segmentation.threshold(
        image, 
        method="otsu"
        )
    
    unique, counts = np.unique(thresh, return_counts=True)
    
    assert len(unique) == 2

def test_morphology(image_binary):
    
    morph = pp.segmentation.morphology(
        image_binary, 
        operation="close", 
        shape="ellipse", 
        kernel_size=4, 
        iterations=2,
        )    
    
    assert not (image_binary==morph).all()

def test_detect_contour(image_binary, image, drawing):
    
    annotations = drawing
    
    pp.segmentation.detect_contour(
        image, 
        )    
    
    pp.segmentation.detect_contour(
        image_binary,
        min_area=1000000,
        )    
    
    annotations = pp.segmentation.detect_contour(
        image_binary, 
        annotations=annotations,
        )    
    
    assert len(annotations) > 0
    
    
def test_edit_contour(image_binary, image, drawing):
    
    annotations = drawing
    
    annotations = pp.segmentation.detect_contour(
        image_binary, 
        annotations=annotations
        )    
    
    annotations = pp.segmentation.edit_contour(
        image, 
        annotations=annotations,
        passive=True,
        )

    assert len(annotations) > 0
    
    
def test_watershed(image_binary, reference_detected, mask_polygon):
    
    annotations = {**reference_detected, **mask_polygon}
    
    annotations = pp.segmentation.detect_contour(
        image_binary, 
        annotations=annotations
        )    
    
    water = pp.segmentation.watershed(
        image_binary, 
        annotations, 
        )

    assert not (image_binary==water).all()    
