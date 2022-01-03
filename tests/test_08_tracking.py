#%% modules

import os
import pytest
import mock

import phenopype as pp


#%% fixtures


@pytest.fixture
def motion_tracker():
    motion_tracker = pp.motion_tracker(pytest.video_path)
    return motion_tracker

@pytest.fixture
def tracking_method():
    fish = pp.tracking_method(label="fish", remove_shadows=True, min_length=30,
                              overlay_colour="red", mode="single",
                              blur=15, # bigger blurring kernel
                              threshold=200 #higher sensitivity
                              )
    isopod = pp.tracking_method(label="isopod", remove_shadows=True, max_length=30,
                                overlay_colour="green", mode="multiple",
                                blur=9, # smaller blurring kernel
                                threshold=180, # lower sensitivity
                                operations=["diameter",  # isopod size
                                            "area",      # isopod area
                                            "grayscale", # isopod pigmentation
                                            "grayscale_background"] # background darkness
                                )
    tracking_method = [fish, isopod]
    return tracking_method

@pytest.fixture
def masks():
    masks = {'mask': {'a': {'info': {'annotation_type': 'mask',
        'phenopype_function': 'create_mask',
        'phenopype_version': '3.0.dev0'},
       'settings': {'tool': 'rectangle',
        'line_width': 1,
        'line_colour': (0, 255, 0),
        'label_size': 1,
        'label_width': 1,
        'label_colour': (0, 255, 0)},
       'data': {'label': 'center',
        'include': True,
        'n': 1,
        'mask': [[(309, 157), (704, 157), (704, 387), (309, 387), (309, 157)]]}},
      'b': {'info': {'annotation_type': 'mask',
        'phenopype_function': 'create_mask',
        'phenopype_version': '3.0.dev0'},
       'settings': {'tool': 'rectangle',
        'line_width': 1,
        'line_colour': (0, 255, 0),
        'label_size': 1,
        'label_width': 1,
        'label_colour': (0, 255, 0)},
       'data': {'label': 'full_arena',
        'include': True,
        'n': 1,
        'mask': [[(181, 54), (839, 54), (839, 494), (181, 494), (181, 54)]]}}}}
    
    return masks

#%% tests

def test_motion_tracker(motion_tracker):
    filepath = motion_tracker.df_image_data["filepath"][0]
    assert filepath == pytest.video_path

def test_video_output(motion_tracker):
    with mock.patch('builtins.input', return_value='y'):
        motion_tracker.video_output(video_format="DIVX", dirpath=pytest.video_out_dir)
    assert os.path.isfile(os.path.join(pytest.video_out_dir, 
                                       os.path.splitext(os.path.basename(pytest.video_path))[0] 
                                       + "_out" 
                                       + os.path.splitext(os.path.basename(pytest.video_path))[1])) 

# def test_create_mask_mt(motion_tracker):
#     test_params = {"flag_test_mode": True,
#                    "flag_tool": "rectangle",
#                    "point_list":[[(189, 52),
#                                      (838, 52),
#                                      (838, 479),
#                                      (189, 479),
#                                      (189, 52)]]}
#     pp.preprocessing.create_mask(motion_tracker, 
#                                  label="full arena",
#                                  tool="rectangle",
#                                  test_params=test_params)
#     test_params = {"flag_test_mode": True,
#                       "flag_tool": "rectangle",
#                       "point_list":[[(314, 149),
#                                      (699, 149), 
#                                      (699, 380), 
#                                      (314, 380), 
#                                      (314, 149)]]}
#     pp.preprocessing.create_mask(motion_tracker, 
#                                  label="center",
#                                  tool="rectangle",
#                                  test_params=test_params)
#     assert len(motion_tracker.df_masks) == 2
    
def test_tracking_method(motion_tracker):
    isopod = pp.tracking_method(
        label="isopod2", 
        remove_shadows=True, 
        max_length=30,
        overlay_colour="green", 
        mode="multiple",
        blur=9, # smaller blurring kernel
        threshold=180, # lower sensitivity
        operations=[
            "diameter",  # isopod size
            "area",      # isopod area
            "grayscale", # isopod pigmentation
            "grayscale_background", # background darkness
            "bgr"
            ])        
    assert len(isopod.__dict__) == 11

def test_run_tracking(motion_tracker, tracking_method, masks):
    
    motion_tracker.detection_settings(
        methods=tracking_method,
        finish_after = 2,
        c_mask=True,
        c_mask_shape="rect",
        c_mask_size=200,
        masks=masks
        )
    
    coordinates = motion_tracker.run_tracking(feedback=False)
    assert len(coordinates) > 0
