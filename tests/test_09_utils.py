#%% modules
import mock
import phenopype as pp

from .settings import image_save_dir, wait_time


#%% tests



def test_load_image_arr(image_path):
    image, df = pp.load_image(image_path, df=True)
    assert all([image.__class__.__name__ == "ndarray",
                df["filename"][0] == "isopods.jpg"])

def test_load_image_ct(image_path):
    image = pp.load_image(image_path, meta=True, cont=True, df=True)
    assert image.__class__.__name__ == "container"

def test_load_meta_data(image_path):
    meta = pp.utils.load_meta_data(image_path, show_fields=True)
    assert meta["ISOSpeedRatings"] == "200"


def test_show_image(image_array):
    test_params = {"flag_test_mode": True,
                   "wait_time": wait_time}
    pp.show_image(image_array, test_params=test_params)
    test_params = {"flag_test_mode": True,
                   "wait_time": wait_time}
    img_list = []
    for i in range(11):
        img_list.append(image_array)
    with mock.patch('builtins.input', return_value='y'):
        pp.show_image(img_list, 
                      test_params=test_params, 
                      position_reset=False)
        
def test_save_image(image_array):
    pp.save_image(image_array, name="test_img", dirpath=image_save_dir)
