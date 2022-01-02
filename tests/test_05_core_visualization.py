#%% modules

import mock
import pytest

import phenopype as pp


#%% fixtures
@pytest.fixture
def canvas(image):
    
    canvas = pp.visualization.select_canvas(
        image, 
        )
    
    return canvas

@pytest.fixture
def container():
    with mock.patch('builtins.input', return_value='y'):
        project = pp.Project(root_dir=pytest.project_root_dir_1)    
        
    project.add_files(
        image_dir=pytest.image_dir, 
        mode="link", 
        include="stickle"
        )
        
    return pp.utils_lowlevel._load_project_image_directory(project.dir_paths[0])

#%% tests

def test_select_canvas(image, image_binary, container):
    
    
    canvas = pp.visualization.select_canvas(
        image_binary, 
        canvas="raw", 
        multi=True,
        )
    
    
    canvas = pp.visualization.select_canvas(
        container, 
        canvas="mod", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        container, 
        canvas="gray", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        container, 
        canvas="red", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        container, 
        canvas="green", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        container, 
        canvas="blue", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        container, 
        canvas="raw", 
        multi=True,
        )
    
    
    canvas = pp.visualization.select_canvas(
        image, 
        canvas="mod", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        image, 
        canvas="gray", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        image, 
        canvas="red", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        image, 
        canvas="green", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        image, 
        canvas="blue", 
        multi=True,
        )
    canvas = pp.visualization.select_canvas(
        image, 
        canvas="raw", 
        multi=True,
        )
    
    assert (image==canvas).all()



def test_draw_contour(canvas, contours):
    
    annotations = contours
    
    canvas_mod = pp.visualization.draw_contour(
        canvas,
        annotations,
        bounding_box=True,
        label=True,
        )
    
    assert not (canvas==canvas_mod).all()



def test_draw_landmarks(canvas, landmarks):
    
    annotations = landmarks
    
    canvas_mod = pp.visualization.draw_landmark(
        canvas,
        annotations,
        )
    
    assert not (canvas==canvas_mod).all()
    
    

def test_draw_mask(canvas, mask_polygon):
    
    annotations = mask_polygon
    
    canvas_mod = pp.visualization.draw_mask(
        canvas,
        annotations,
        label=True,
        )
    
    assert not (canvas==canvas_mod).all()
    
    
    
def test_draw_polyline(canvas, polyline):
    
    annotations = polyline
    
    canvas_mod = pp.visualization.draw_polyline(
        canvas,
        annotations,
        )
    
    assert not (canvas==canvas_mod).all()



def test_draw_reference(canvas, reference_created, reference_detected):
    
    annotations = reference_created 
    
    canvas_mod = pp.visualization.draw_reference(
        canvas,
        annotations,
        label=True,
        )
    
    annotations = reference_detected 
        
    canvas_mod = pp.visualization.draw_reference(
        canvas,
        annotations,
        label=True,
        )
    
    assert not (canvas==canvas_mod).all()



