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
    

def test_compute_shape_feature(image_binary):
    
    annotations = pp.segmentation.detect_contour(
        image_binary, 
        )    
    
    annotations = pp.measurement.compute_shape_features(
        annotations,
        features=["basic","moments","hu_moments"],
)
    
    assert len(annotations) > 0
    
def test_detect_skeleton(image_binary):
    
    annotations = pp.segmentation.detect_contour(
        image_binary, 
        )    
    
    annotations = pp.measurement.detect_skeleton(
        annotations,
        )
    
    assert len(annotations) > 0
    
def test_compute_texture_feature(image, image_binary):
    
    annotations = pp.segmentation.detect_contour(
        image_binary, 
        )    
    
    annotations = pp.measurement.compute_texture_features(
        image,
        annotations,
        features=["firstorder", "shape", "glcm", "gldm", "glrlm", "glszm", "ngtdm"],
    )
    
    assert len(annotations) > 0
        
