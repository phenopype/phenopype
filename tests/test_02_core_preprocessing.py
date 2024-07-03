#%% modules

import pytest
import numpy as np

import phenopype as pp

#%% tests

def test_blur(image):
    
    image_blurred = pp.preprocessing.blur(image)
    image_blurred = pp.preprocessing.blur(image, kernel_size=4)
    image_blurred = pp.preprocessing.blur(image, method="gaussian")
    image_blurred = pp.preprocessing.blur(image, method="median")
    image_blurred = pp.preprocessing.blur(image, method="bilateral")

    assert not (image == image_blurred).all()

def test_create_mask(image, mask_polygon):
    
    annotations = mask_polygon

    annotations = pp.preprocessing.create_mask(
        image, 
        annotations=annotations, 
        annotation_id="a", 
        interactive=False,
        )

    assert len(annotations) > 0
    
    
def test_detect_mask(image):
    
    annotations = pp.preprocessing.detect_mask(
        image, 
        circle_args={"param1": 150, "param2": 150, "min_radius":1000}, 
        resize=0.5
        )
    
    annotations = pp.preprocessing.detect_mask(
        image, 
        circle_args={"param1": 150, "param2": 150, "max_radius":150}, 
        resize=0.5
        )

    assert len(annotations) > 0
    
def test_create_reference(reference_created, settings):
    
    image = pp.load_image(pytest.reference_image_path)
    
    annotations = reference_created
    
    annotations = pp.preprocessing.create_reference(
        image,
        annotations=annotations, 
        interactive=False
        )
    
    assert len(annotations) > 0
    
def test_detect_reference(image, reference_created, settings):
    
    annotations = reference_created
    
    reference_image = pp.load_image(pytest.reference_image_path)
    
    coords = annotations["reference"]["a"]["data"]["mask"][0]
    template = reference_image[coords[0][1]:coords[2][1], coords[0][0]:coords[1][0]]
    px_ratio = annotations["reference"]["a"]["data"]["reference"][0]
    unit = annotations["reference"]["a"]["data"]["reference"][1]
    
    black_dummy = np.zeros((5001, 5001, 3), dtype=np.uint8)
    
    annotations = pp.preprocessing.detect_reference(
        image=black_dummy,
        template=template,
        correct_colours=True,
        px_ratio=px_ratio,
        unit=unit
        )
    
    annotations = pp.preprocessing.detect_reference(
        image=image,
        template=template,
        resize=0.25,
        correct_colours=True,
        px_ratio=px_ratio,
        unit=unit
        )
        
    assert len(annotations) > 0

    
def test_decompose_image(image):
    
    mod = pp.preprocessing.decompose_image(image, "bier")
    mod = pp.preprocessing.decompose_image(image, "raw")
    mod = pp.preprocessing.decompose_image(image, "hsv")
    mod = pp.preprocessing.decompose_image(image, "hsv", channels=0)
    mod = pp.preprocessing.decompose_image(image, "hsv", channels=[1,2])
    mod = pp.preprocessing.decompose_image(image, channel="blue")
    mod = pp.preprocessing.decompose_image(image, "rgb", channel="blue")

    canvas = pp.visualization.select_canvas(mod, multi_channel=True)

    assert not (image == canvas).all()
    
    
def test_write_comment(image, comment):
    
    annotations = comment
    
    annotations = pp.preprocessing.write_comment(
        image=image,
        annotations=annotations,
        interactive=False,
        )

    assert len(annotations) > 0



