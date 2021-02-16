#%% modules

import phenopype as pp

from .settings import flag_overwrite


#%% tests

def test_landmarks(project_container):
    test_params = {"flag_test_mode": True,
                    "points":[(676, 391),
                              (674, 497),
                              (775, 415),
                              (897, 381),
                              (830, 521),
                              (986, 278),
                              (964, 530),
                              (1188, 254),
                              (1219, 540),
                              (1296, 233),
                              (1339, 240),
                              (1466, 249),
                              (1317, 458),
                              (1353, 583),
                              (1358, 502),
                              (1447, 502)] }
    pp.measurement.landmarks(project_container, 
                             point_size=10, 
                             overwrite=flag_overwrite,
                             test_params=test_params)
    assert len(project_container.df_landmarks) == len(test_params["points"])
    
def test_colour_intensity(project_container):
    pp.measurement.colour_intensity("string_dummy")
    pp.measurement.colour_intensity(project_container.df_contours)
    pp.measurement.colour_intensity(project_container, channels=["gray"])
    assert hasattr(project_container, "df_colours")
    
def test_shape_features(project_container):
    pp.measurement.shape_features(project_container)
    assert "perimeter_length" in project_container.df_shapes

def test_polylines(project_container):
    test_params = {"flag_test_mode": True,
                   "point_list":[[(1070, 307),
                                  (1216, 319),
                                  (1336, 321),
                                  (1495, 357),
                                  (1689, 391),
                                  (1869, 406),
                                  (1994, 401),
                                  (2059, 391)]]}
    pp.measurement.polylines(project_container, 
                             line_width=5, 
                             overwrite=flag_overwrite,
                             test_params=test_params)
    assert len(project_container.df_polylines) == len(test_params["point_list"][0])

def test_skeletonize(project_container):
    pp.measurement.skeletonize(project_container)
    assert "skeleton_coords" in project_container.df_contours