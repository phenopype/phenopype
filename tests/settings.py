# %% testing setup

root = "../../phenopype-tutorials"

image_filepath = root + r"tutorials/images/isopods.jpg"
image_ref_filepath = root + r"tutorials/images/stickle1.JPG"
image_dir = r"tutorials/images"
video_path = r"tutorials/images/isopods_fish.mp4"

image_save_dir = r"_temp/tests/images"
root_dir1 = r"_temp/tests/project1"
root_dir2 = r"_temp/tests/project2"
video_out_dir = r"_temp/tests/video"

pype_name = "v1"
template_ex1 = "ex1"

template_test1 = "tests/test_templates/test1.yaml"
template_test2 = "tests/test_templates/test2.yaml"

stickle_image = 3
wait_time = 100
flag_overwrite = True

config_target = \
"""- threshold:"""
config_replacement = \
"""- threshold:
      value: 127"""