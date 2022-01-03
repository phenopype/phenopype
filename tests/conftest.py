import os
import mock
import pytest

import phenopype as pp

#%% variables

@pytest.fixture(scope="session")
def settings():
    
    ## general 
    test_dir = "_temp/tests"
    tutorial_dir = "../phenopype-tutorials/"
    
    pytest.test_dir = test_dir
    pytest.tutorial_dir = tutorial_dir
    
    
    ## project
    pytest.project_root_dir_1 =  os.path.join(test_dir, "project1")
    pytest.project_root_dir_2 =  os.path.join(test_dir, "project2")
    
    pytest.image_dir = os.path.join(tutorial_dir, "tutorials/images")
    pytest.reference_image_path = os.path.join(tutorial_dir, "tutorials/images", "stickleback_top.jpg")

    pytest.tag_1 = "v1"
    pytest.tag_2 = "v2"
    
    pytest.template_path_1 = "tests/templates/test1.yaml"
    pytest.template_path_2 = "tests/templates/test2.yaml"
    pytest.template_path_3 = "tests/templates/test3.yaml"

    pytest.edit_config_target = \
    """        - create_mask:
            tool: polygon"""
                
    pytest.edit_config_replacement = \
    """        - create_mask:
            tool: rectangle"""
            
    ## single image
    pytest.image_path = tutorial_dir + r"tutorials/images/stickle1.jpg"

    ## video
    pytest.video_path = os.path.join(tutorial_dir, r"tutorials/videos/isopods_fish.mp4")
    pytest.video_out_dir =  os.path.join(test_dir, "video")
    
    
    


#%% project
    
@pytest.fixture(scope="session")
def project():
    with mock.patch('builtins.input', return_value="y"):
        test_project = pp.Project(root_dir=pytest.project_root_dir_1)
    return test_project
    

@pytest.fixture(scope="session")
def image():
    return pp.load_image(pytest.image_path)

    
@pytest.fixture(scope="session")
def mask_polygon():

    return {'mask': {'a': {'info': {'annotation_type': 'mask',
        'phenopype_function': 'create_mask',
        'phenopype_version': '3.0.dev0'},
       'settings': {'tool': 'polygon',
        'line_width': 5,
        'line_colour': (0, 255, 0),
        'label_size': 1,
        'label_width': 1,
        'label_colour': (0, 255, 0)},
       'data': {'label': "armour-plates",
        'include': True,
        'n': 1,
        'mask': [[(1377, 273),
          (1380, 444),
          (1816, 424),
          (1814, 382),
          (1682, 357),
          (1521, 290),
          (1394, 274),
          (1377, 273)]]}}}}

@pytest.fixture(scope="session")
def image_binary(image,  mask_polygon):
        
    annotations = mask_polygon
        
    image_blurred = pp.preprocessing.blur(image, kernel_size=7)
        
    return pp.segmentation.threshold(
        image_blurred, 
        method="adaptive", 
        blocksize=199, 
        constant=10, 
        channel="red",
        annotations=annotations,
        verbose=False,
        )


@pytest.fixture(scope="session")
def contours(image_binary):

    return pp.segmentation.detect_contour(
        image_binary, 
        )    


        
@pytest.fixture(scope="session")
def reference_created():

    return {'reference': {'a': {'info': {'annotation_type': 'reference',
        'phenopype_function': 'create_reference',
        'phenopype_version': '3.0.dev0'},
       'settings': {},
       'data': {'label': None,
        'reference': (36.12, 'mm'),
        'support': [(1843, 1753), (2204, 1765)],
        'mask': [[(1665, 1111),
          (2517, 1111),
          (2517, 1957),
          (1665, 1957),
          (1665, 1111)]]}}}}
    
@pytest.fixture(scope="session")
def reference_detected():
    
    return {'reference': {'a': {'info': {'annotation_type': 'reference',
        'phenopype_function': 'detect_reference',
        'phenopype_version': '3.0.dev0'},
       'settings': {'get_mask': True,
        'correct_colours': True,
        'min_matches': 10,
        'resize': 1},
       'data': {'reference': (34.869, 'mm'),
        'mask': [[(1202, 1555),
          (1247, 740),
          (380, 694),
          (335, 1508),
          (1202, 1555)]]}}}}

@pytest.fixture(scope="session")
def landmarks():

    return {'landmark': {'a': {'info': {'annotation_type': 'landmark',
        'phenopype_function': 'set_landmark',
        'phenopype_version': '3.0.dev0'},
       'settings': {'point_size': 5,
        'point_colour': (0, 255, 0),
        'label': True,
        'label_size': 1,
        'label_width': 1,
        'label_colour': (0, 255, 0)},
       'data': {'landmark': [(840, 369),
         (1183, 293),
         (1675, 309),
         (1740, 442),
         (1238, 458),
         (832, 437)]}}}}


@pytest.fixture(scope="session")
def polyline():
    
    return {'line': {'a': {'info': {'phenopype_function': 'set_polyline',
        'phenopype_version': '3.0.dev0',
        'annotation_type': 'line'},
       'settings': {'line_width': 5, 'line_colour': (0, 255, 0)},
       'data': {'line': [[(746, 706),
          (1024, 557),
          (1238, 747),
          (1560, 528),
          (1821, 706),
          (2092, 571)],
         [(924, 694), (1180, 869), (1574, 641), (1893, 814), (2119, 694)]]}}}}
    

@pytest.fixture(scope="session")
def comment():
    
    return {'comment': {'a': {'info': {'phenopype_function': 'write_comment',
        'phenopype_version': '3.0.dev0',
        'annotation_type': 'comment'},
       'settings': {},
       'data': {'label': 'test msg', 'comment': 'THIS IS A TEST'}}}}