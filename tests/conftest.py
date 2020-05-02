#%% modules

import os
import mock
import pytest
    
import phenopype as pp

from .settings import root_dir2, image_dir, pype_name, preset, ref_image, stickle_image, flag_overwrite

#%% project


@pytest.fixture(scope="session")
def project_container():
    with mock.patch('builtins.input', return_value='y'):
        proj = pp.project(root_dir=root_dir2, overwrite=flag_overwrite)
    proj.add_files(image_dir=image_dir, 
                      raw_mode="copy", 
                      include="stickle")
    proj.add_config(name=pype_name, config_preset=preset)
    pp.project.save(proj)
    proj = pp.project.load(root_dir2)
    obj_input = pp.load_directory(proj.dirpaths[0])
    project_container = obj_input
    return obj_input ## container

# @pytest.fixture(scope="session")
# def project_directory():
#     if not os.getcwd() == r"/home/travis/build/mluerig/phenopype":
#         os.chdir(r"E:\git_repos\phenopype")
#     if os.path.isfile(os.path.join(root_dir2, "project.data")):
#         project = pp.project.load(root_dir2)
#     else: 
#         with mock.patch('builtins.input', return_value='y'):
#             project = pp.project(root_dir=root_dir2, overwrite=flag_overwrite)
#         project.add_files(image_dir=image_dir, 
#                           raw_mode="link", 
#                           include="stickle")
#         project.add_config(name=pype_name, config_preset=preset)
#         project.add_scale(reference_image=ref_image, template=True)    
#         pp.project.save(project)
#     directory = os.path.join(project.root_dir, 
#                              project.dirpaths[stickle_image])
#     return directory