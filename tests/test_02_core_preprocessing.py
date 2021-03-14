#%% modules

import phenopype as pp

from .settings import flag_overwrite

#%% tests

def test_create_mask(project_container):
    test_params = {"flag_test_mode": True,
              "flag_tool": "polygon",
              "point_list":[[(1376, 272),
                             (1378, 435),
                             (1814, 422),
                             (1816, 379),
                             (1757, 377),
                             (1627, 336),
                             (1504, 295),
                             (1389, 275),
                             (1376, 272)]] }
    pp.preprocessing.create_mask(project_container, 
                                 tool="rect", 
                                 overwrite=flag_overwrite,
                                 test_params=test_params)
    assert len(project_container.df_masks) > 0
    
def test_invert_image(project_container):
    inv = pp.preprocessing.invert_image(project_container.image)
    assert not (project_container.image==inv).all()
    
def test_resize_image(project_container):
    res = pp.preprocessing.resize_image(project_container.image, factor=0.9)
    assert project_container.image.shape>=res.shape

def test_enter_data(project_container):
    test_params = {"flag_test_mode": True,
              "entry": "142501"}
    pp.preprocessing.enter_data(project_container, 
                                columns="ID", 
                                overwrite=flag_overwrite,
                                test_params=test_params)
    assert project_container.df_image_data.iloc[0]["ID"] == "142501"

def test_detect_reference(project_container):
    pp.preprocessing.detect_reference(project_container, 
                                      overwrite=flag_overwrite,
                                      equalize=True)
    assert project_container.reference_detected_px_mm_ratio == 35.4

def test_create_reference(project_container):
    test_params = {"flag_test_mode": True,
              "flag_tool": "reference",
              "reference_coords": [(701, 741), 
                               (1053, 774)],
              "point_list": [[(316, 675), 
                              (1236, 675), 
                              (1236, 1549), 
                              (316, 1549), 
                              (316, 675)]],
              "rect_list": [[316, 675, 1236, 1549]],
              "entry": "10"}
    pp.preprocessing.create_reference(project_container, 
                                  template=False, 
                                  overwrite=flag_overwrite,
                                  test_params=test_params)
    assert project_container.reference_manually_measured_px_mm_ratio == 35.35434909597403



