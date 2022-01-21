#%% modules
import mock
import os
import pytest 

import phenopype as pp

# from .settings import pype_name


#%% tests

def test_save_canvas(image, settings):

    pp.export.save_canvas(
        image, 
        dir_path=pytest.test_dir,
        )
    
    pp.export.save_canvas(
        image, 
        dir_path=pytest.test_dir,
        ext="jpg",
        overwrite=True,
        resize=1,
        )

    assert os.path.isfile(os.path.join(pytest.test_dir, "canvas.jpg"))


def test_save_annotation(contours, mask_polygon, settings):

    annotations = {**contours, **mask_polygon}
    
    pp.export.save_annotation(
        annotations, 
        dir_path=pytest.test_dir,
        )
    
    pp.export.save_annotation(
        annotations, 
        dir_path=pytest.test_dir,
        )
    
    pp.export.save_annotation(
        annotations, 
        dir_path=pytest.test_dir,
        )
    
    pp.export.save_annotation(
        annotations, 
        dir_path=pytest.test_dir,
        overwrite=True,
        )
    
    pp.export.save_annotation(
        annotations, 
        dir_path=pytest.test_dir,
        overwrite="file",
        )

    assert os.path.isfile(os.path.join(pytest.test_dir, "annotations.json"))
    
    
def test_load_annotation(settings, image):

    path = os.path.join(pytest.test_dir, "annotations.json")
    
    
    annotations = pp.export.load_annotation(
        path, 
        annotation_type="comment",
        annotation_id="a",
        )
    
    annotations = pp.export.load_annotation(
        path, 
        annotation_type="comment",
        annotation_id=["a"],
        )
    
        
    annotations = pp.export.load_annotation(
        path, 
        annotation_type=["comment","mask"]
        )


    annotations = pp.export.load_annotation(
        path, 
        )
    
    ## check if coords are legible
    canvas = pp.visualization.select_canvas(
        image, 
        )
    
    canvas_mod = pp.visualization.draw_contour(
        canvas,
        annotations,
        )
    
    annotations = pp.preprocessing.create_mask(
        image, 
        annotations=annotations, 
        passive=True,
        )

    assert not (image==canvas_mod).all() and len(annotations) > 0

    
def test_save_ROI(image, mask_polygon, settings):
    
    annotations = mask_polygon
    
    save_dir = os.path.join(pytest.test_dir,"ROI")

    os.mkdir(save_dir)
    
    pp.export.save_ROI(
        image, 
        file_name="test",
        annotations=annotations,
        dir_path=save_dir,
        )

    assert len(os.listdir(save_dir)) == len(annotations["mask"]["a"]["data"]["mask"])
    
    
def test_export_csv(image, contours, comment, settings):
    
    annotations = {**contours, **comment}
    
    annotations = pp.measurement.compute_texture_features(
        image,
        annotations,
        features=["firstorder", "shape", "glcm", "gldm", "glrlm", "glszm", "ngtdm"],
    )

    annotations = pp.measurement.compute_shape_features(
        annotations=annotations,
        features=["basic","moments","hu_moments"],
    )
    
    save_dir = pytest.test_dir

    pp.export.export_csv(
        annotations=annotations,
        annotation_type = "comment",
        dir_path=save_dir,
        )

    pp.export.export_csv(
        annotations=annotations,
        dir_path=save_dir,
        )

    assert os.path.isfile(os.path.join(save_dir, "texture_features.csv"))
