# %% testing setup
import os

root_dir1 = "tests/resources/project1"
root_dir2 = "tests/resources/project2"
image_dir = "tutorials/images"
pype_name = "v1"
ref_image = "0__stickleback_side"
preset = "demo1"

stickle_image = 3

flag_overwrite = False
if os.getcwd() == "/home/travis/build/mluerig/phenopype":
    flag_feedback = False
else:
    flag_feedback = True