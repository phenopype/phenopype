#%% modules

import os
import mock
import pytest
import shutil 
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

import phenopype as pp

#%% variables

# os.chdir(r"D:\git-repos\phenopype\phenopype")

@pytest.fixture(scope="session")
def settings():
    
    pytest.tutorials_url = "https://github.com/phenopype/phenopype-tutorials/archive/refs/heads/main.zip"
    
    ## general 
    test_dir = r"_temp/tests"
    image_dir = r"_temp/tests/phenopype-tutorials-main/tutorials/data"
    video_dir = r"_temp/tests/phenopype-tutorials-main/tutorials/data"

    pytest.test_dir = test_dir
    pytest.image_dir = image_dir
        
    ## project
    pytest.project_root_dir_1 =  os.path.join(test_dir, "project1")
    pytest.project_root_dir_2 =  os.path.join(test_dir, "project2")
    
    pytest.reference_image_path = os.path.join(image_dir, "stickleback_top.jpg")

    pytest.tag_1 = "v1"
    pytest.tag_2 = "v2"
    
    pytest.template_path_1 = "tests/templates/test1.yaml"
    pytest.template_path_2 = "tests/templates/test2.yaml"
    pytest.template_path_3 = "tests/templates/test3.yaml"
    pytest.template_path_4 = "tests/templates/test4.yaml"


    pytest.edit_config_target = \
    """        - create_mask:
            tool: polygon"""
                
    pytest.edit_config_replacement = \
    """        - create_mask:
            tool: rectangle"""
            
    ## single image
    pytest.image_path =  os.path.join(image_dir, r"stickle1.jpg")

    ## video
    pytest.video_path = os.path.join(video_dir, r"isopods_fish.mp4")
    pytest.video_out_dir =  test_dir
    
    
def pytest_configure(config):
    
    test_dir = r"_temp/tests"
    tutorials_url = "https://github.com/phenopype/phenopype-tutorials/archive/refs/heads/main.zip" 
    
    if os.path.isdir(test_dir):
       shutil.rmtree(test_dir) 
       print("Removed existing test dir {}".format(os.path.abspath(test_dir)))
       
    os.makedirs(test_dir)
    
    http_response = urlopen(tutorials_url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=test_dir)
    

#%% project
    
@pytest.fixture(scope="session")
def project():
    with mock.patch('builtins.input', return_value="y"):
        
        test_project = pp.Project(root_dir=pytest.project_root_dir_1)
        
    return test_project
    

@pytest.fixture(scope="session")
def image():
    
    image = pp.load_image(pytest.image_path)
    
    return image

    
@pytest.fixture(scope="session")
def mask_polygon():

    mask_polygon = {'mask': {'a': {'info': {'annotation_type': 'mask',
        'phenopype_function': 'create_mask',
        'phenopype_version': '3.0.dev0'},
       'settings': {'tool': 'polygon',
        'line_width': 5,
        'line_colour': (0, 255, 0),
        'label_size': 1,
        'label_width': 1,
        'label_colour': (0, 255, 0)},
       'data': {'label': "armor-plates",
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
    
    return mask_polygon

@pytest.fixture(scope="session")
def image_binary(image,  mask_polygon):
        
    annotations = mask_polygon
        
    image_blurred = pp.preprocessing.blur(image, kernel_size=7)
    image_binary = pp.segmentation.threshold(
            image=image_blurred, 
            method="adaptive", 
            blocksize=199, 
            constant=10, 
            channel="green",
            annotations=annotations,
            verbose=True,
            )
           
    return image_binary


@pytest.fixture(scope="session")
def contours(image_binary):
    
    contours = pp.segmentation.detect_contour(
        image_binary, 
        )    

    return contours


        
@pytest.fixture(scope="session")
def reference_created():
    
    reference_created = {'reference': {'a': {'info': {'annotation_type': 'reference',
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

    return reference_created
    
@pytest.fixture(scope="session")
def reference_detected():
    
    reference_detected = {'reference': {'a': {'info': {'annotation_type': 'reference',
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
    
    return reference_detected

@pytest.fixture(scope="session")
def landmarks():
    
    landmarks = {'landmark': {'a': {'info': {'annotation_type': 'landmark',
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

    return landmarks


@pytest.fixture(scope="session")
def polyline():
    
    polyline =  {'line': {'a': {'info': {'phenopype_function': 'set_polyline',
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
    
    return polyline
    

@pytest.fixture(scope="session")
def comment():
    
    comment = {'comment': {'a': {'info': {'phenopype_function': 'write_comment',
        'phenopype_version': '3.0.dev0',
        'annotation_type': 'comment'},
       'settings': {},
       'data': {'label': 'test msg', 'comment': 'THIS IS A TEST'}}}}
    
    return comment


@pytest.fixture
def drawing():

    drawing = {"drawing": {'a': {'info': {'annotation_type': 'drawing',
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
    
    return drawing