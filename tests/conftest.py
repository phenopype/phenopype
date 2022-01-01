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
    pytest.project_root_dir_3 =  os.path.join(test_dir, "project3")
    
    pytest.image_dir = os.path.join(tutorial_dir, "tutorials/images")
    pytest.reference_image_path = os.path.join(tutorial_dir, "tutorials/images", "stickleback_top.jpg")

    pytest.tag_1 = "v1"
    
    pytest.template_path_1 = "tests/test_templates/test1.yaml"
    pytest.template_path_2 = "tests/test_templates/test2.yaml"

    pytest.edit_config_target = \
    """        - create_mask:
            tool: polygon"""
                
    pytest.edit_config_replacement = \
    """        - create_mask:
            tool: rectangle"""
            
            
    ## various
    pytest.image_path = tutorial_dir + r"tutorials/images/stickle1.jpg"


    # pytest.video_path = r"tutorials/images/isopods_fish.mp4"
    
    # pytest.image_save_dir = test_dir + r"/images"
    # pytest.video_out_dir =  test_dir + r"/video"
    
    # pytest.template_ex1 = "ex1"
    
    


#%% project
    
@pytest.fixture(scope="session")
def project(settings):
    with mock.patch('builtins.input', return_value="y"):
        test_project = pp.Project(root_dir=pytest.project_root_dir_1)
    return test_project
    

@pytest.fixture(scope="session")
def image(settings):
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
       'data': {'label': None,
        'include': True,
        'n': 1,
        'mask': [[(1340, 289),
          (1346, 444),
          (1772, 420),
          (1775, 371),
          (1563, 307),
          (1421, 277),
          (1360, 274),
          (1340, 289)]]}}}}

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



# @pytest.fixture(scope="session")
# def project_container():
#     with mock.patch('builtins.input', return_value='y'):
#         proj = pp.Project(root_dir=root_dir_2, overwrite=flag_overwrite)
#     proj.add_files(image_dir=image_dir, 
#                       raw_mode="copy", 
#                       include="stickle")
#     # proj.add_config(name=pype_name, template=template_path_1)
#     pp.project.save(proj)

#     # proj.add_reference(name="ref1", reference_image=0, template=True, test_params=ref_params)
#     obj_input = pp.load_directory(proj.dirpaths[0], save_suffix=tag_1)
#     obj_input.load()
#     return obj_input 

# @pytest.fixture(scope="session")
# def project_directory():
#     proj = pp.project.load(root_dir_2)
#     image = proj.dirpaths[0]
#     # project_directory = image 
#     return image

# @pytest.fixture(scope="session")
# def motion_tracker():
#     mt = pp.motion_tracker(video_path)
#     motion_tracker = mt
#     return mt

# @pytest.fixture(scope="session")
# def tracking_method():
#     fish = pp.tracking_method(label="fish", remove_shadows=True, min_length=30,
#                               overlay_colour="red", mode="single",
#                               blur=15, # bigger blurring kernel
#                               threshold=200 #higher sensitivity
#                              )
#     isopod = pp.tracking_method(label="isopod", remove_shadows=True, max_length=30,
#                                 overlay_colour="green", mode="multiple",
#                                 blur=9, # smaller blurring kernel
#                                 threshold=180, # lower sensitivity
#                                 operations=["diameter",  # isopod size
#                                             "area",      # isopod area
#                                             "grayscale", # isopod pigmentation
#                                             "grayscale_background"] # background darkness
#                                )
#     methods = [fish, isopod]
#     tracking_method = methods
#     return methods

# @pytest.fixture(scope="session")
# def image_path():
#     return image_filepath

# @pytest.fixture(scope="session")
# def image_array(image_path):
#     if os.path.isdir(image_save_dir):
#         shutil.rmtree(image_save_dir) 
#     with mock.patch('builtins.input', return_value='y'):
#         img = pp.load_image(image_path, dirpath=image_save_dir)
#     image = img
#     return img
