#%% modules

import phenopype as pp

#%% tests

def test_set_landmark(image, landmarks):
    
    annotations = landmarks

    annotations = pp.measurement.set_landmark(
        image, 
        annotations=annotations,
        interactive=False,
        ) 
    
    assert len(annotations) > 0


def test_set_polyline(image, polyline):
    
    annotations = polyline

    annotations = pp.measurement.set_polyline(
        image, 
        annotations=annotations,
        interactive=False,
        ) 
    
    assert len(annotations) > 0
    

def test_compute_shape_moments(image_binary):
    
    annotations = pp.segmentation.detect_contour(
        image_binary, 
        )    
    
    annotations = pp.measurement.compute_shape_moments(
        annotations,
        features=["basic","moments","hu_moments"],
)
    
    assert len(annotations) > 0
    
# def test_detect_skeleton(image_binary):
    
#     annotations = pp.segmentation.detect_contour(
#         image_binary, 
#         )    
    
#     annotations = pp.measurement.detect_skeleton(
#         annotations,
#         )
    
#     assert len(annotations) > 0
    
        
