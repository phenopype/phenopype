#%% modules

import os
import cv2
import pytest
import mock
import random

import phenopype as pp

from .settings import video_path, video_out_dir





#%% tests

def test_motion_tracker(motion_tracker):
    filepath = motion_tracker.df_image_data["filepath"][0]
    assert filepath == video_path

def test_video_output(motion_tracker):
    with mock.patch('builtins.input', return_value='y'):
        motion_tracker.video_output(video_format="DIVX", dirpath=video_out_dir)
    assert os.path.isfile(os.path.join(video_out_dir, 
                                       os.path.splitext(os.path.basename(video_path))[0] 
                                       + "_out" 
                                       + os.path.splitext(os.path.basename(video_path))[1])) 

def test_create_mask_mt(motion_tracker):
    test_params = {"flag_test_mode": True,
                      "flag_tool": "rectangle",
                      "point_list":[[(189, 52),
                                     (838, 52),
                                     (838, 479),
                                     (189, 479),
                                     (189, 52)]]}
    pp.preprocessing.create_mask(motion_tracker, 
                                 label="full arena",
                                 tool="rectangle",
                                 test_params=test_params)
    test_params = {"flag_test_mode": True,
                      "flag_tool": "rectangle",
                      "point_list":[[(314, 149),
                                     (699, 149), 
                                     (699, 380), 
                                     (314, 380), 
                                     (314, 149)]]}
    pp.preprocessing.create_mask(motion_tracker, 
                                 label="center",
                                 tool="rectangle",
                                 test_params=test_params)
    assert len(motion_tracker.df_masks) == 2
    
def test_tracking_method(motion_tracker):
    isopod = pp.tracking_method(label="isopod", remove_shadows=True, max_length=30,
                            overlay_colour="green", mode="multiple",
                            blur=9, # smaller blurring kernel
                            threshold=180, # lower sensitivity
                            operations=["diameter",  # isopod size
                                        "area",      # isopod area
                                        "grayscale", # isopod pigmentation
                                        "grayscale_background", # background darkness
                                        "bgr"]
                           )        
    assert len(isopod.__dict__) == 11

def test_detection_settings(motion_tracker, tracking_method):
    motion_tracker.detection_settings(methods=tracking_method,
                         finish_after = 2,
                         c_mask=True,
                         c_mask_shape="rect",
                         c_mask_size=200)
    assert len(motion_tracker.methods) == 2

def test_run_tracking(motion_tracker):
    coordinates = motion_tracker.run_tracking()
    assert len(coordinates) > 0
