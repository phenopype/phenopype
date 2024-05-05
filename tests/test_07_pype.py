#%% modules

import copy
import mock
import os
import pytest

import phenopype as pp


@pytest.fixture
def all_annotations(comment, mask_polygon, reference_detected, polyline, landmarks):
    
    all_annotations = {**comment, **mask_polygon, **reference_detected, **polyline, **landmarks}

    return all_annotations

#%% tests


def test_pype(settings, all_annotations, reference_detected):
    
    annotations = all_annotations
    
    with mock.patch('builtins.input', return_value='y'):
        project = pp.Project(root_dir=pytest.project_root_dir_2)
        
    project.add_files(
        image_dir=pytest.image_dir, 
        mode="link", 
        include="stickle"
        )
    
    project.add_reference_template(
        image_path=pytest.reference_image_path,
        reference_id="ref1",
        interactive=False,
        overwrite=True,
        annotations=annotations,
        )
    
    project.add_config(
        tag=pytest.tag_1, 
        template_path=pytest.template_path_2
        )
    
    project.add_config(
        tag=pytest.tag_2, 
        template_path=pytest.template_path_3
        )
    
    pp.export.save_annotation(
        annotations, 
        dir_path=project.dir_paths[0],
        file_name="annotations_" + pytest.tag_1 + ".json",
        )

    p = pp.Pype(project.dir_paths[0], pytest.tag_1, feedback=False, interactive=False, visualize=False)
    p = pp.Pype(project.dir_paths[0], pytest.tag_2, feedback=False, interactive=False, visualize=False)

    assert p.__class__.__name__ == "Pype"
    
    