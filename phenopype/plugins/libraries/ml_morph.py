#%% imports

clean_namespace = dir()

import copy
import cv2
import json
import numpy as np
import pandas as pd
import os
import random
import shutil

from dataclasses import make_dataclass

from phenopype import __version__
from phenopype import _config
from phenopype import core
from phenopype import main
from phenopype import settings
from phenopype import utils
from phenopype import utils_lowlevel

import phenomorph as ml_morph

import xml.etree.ElementTree as ET
from xml.dom import minidom
import dlib



#%% namespace cleanup

funs = ['ProjectLink']

def __dir__():
    return clean_namespace + funs


#%% classes

project = []


# class ProjectLink:

#     def __init__(
#             self, 
#             projects, 
#             root_dir,
#             tag=None,
#             load=True,
#             overwrite=False,
#             **kwargs,
#             ):
        
#         # =============================================================================
#         # setup
        
#         ## create flags
#         flags = make_dataclass(
#             cls_name="flags", fields=[
#                 ("overwrite", bool, overwrite),
#                 ("load", bool, load),
#                 ]
#         )
        
#         ## list check and attach projects
#         self.projects = {}
#         if not projects.__class__.__name__ == "list":
#             projects = [projects]
                
#         for project in projects:
#             if project.__class__.__name__ == "str":
#                 if os.path.isdir(project):
#                     project = main.Project(project)
#                 else:
#                     print("wrong directory path - couldn't find {}".format(project))
#             project_name = os.path.basename(project.root_dir)    
#             self.projects[project_name] = project

#         ## multi-check and make root dir
#         if len(self.projects) > 1:
#             self.root_dir = copy.deepcopy(root_dir)
#         else:
#             self.root_dir = copy.deepcopy(os.path.join(root_dir, "ml_morph"))       
                        
#         ## make other directories
#         self.image_dir = os.path.join(self.root_dir, "images")
#         self.model_dir = os.path.join(self.root_dir, "models")
#         self.config_dir = os.path.join(self.root_dir, "config")
#         self.xml_dir = os.path.join(self.root_dir, "xml")
#         for directory in [self.root_dir, self.image_dir, self.config_dir, self.xml_dir]:
#             if not os.path.exists(directory):
#                 os.makedirs(directory)
                
#         ## attach model
#         self.model = ml_morph.model.Model(self.root_dir)
#         if len(self.projects) > 1:            
#             print("- {} projects loaded ({})".format(len(self.projects), ', '.join(list(self.projects.keys()))))

#         ## load existing components
#         if not tag.__class__.__name__ == "NoneType":
#             ret = utils_lowlevel._file_walker(self.xml_dir, include=[tag], pype_mode=True)[0]
#             if len(ret) > 0:
#                 print("- found training and test datasets \"test_{}.xml\" and \"train_{}.xml\"".format(tag, tag))
#             config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))
#             if os.path.isfile(config_path):
#                 self.config_path = config_path
#                 print("- loaded config \"config_{}.yaml\"".format(tag))
#             model_path = os.path.join(self.model_dir, "predictor_{}.dat".format(tag))
#             if os.path.isfile(model_path):
#                 self.model_path = model_path
#                 print("- loaded model \"predictor_{}.dat\"".format(tag))

#     def create_training_data(
#             self,
#             tag,
#             mode="link",
#             project_tag=None,
#             overwrite=False,
#             landmark_id=None,
#             mask=False,
#             mask_id=None,
#             flip=False,
#             random_seed=42,
#             split=0.8,
#             prop_train=None,
#             prop_test=None,
#             n_train=None,
#             n_test=None,
#             parameters=None,
#             ):
    
#         # =============================================================================
#         # setup
    
#         ## define flags
#         flags = make_dataclass(
#             cls_name="flags", fields=[
#                 ("overwrite", bool, overwrite),
#                 ]
#         )
    
#         annotation_type = settings._landmark_type
#         annotation_id = landmark_id
        
#         ## overwrite check
#         ret = utils_lowlevel._file_walker(self.xml_dir, include=[tag], pype_mode=True)[0]
#         if len(ret) > 0 and not flags.overwrite:
#             print("test_{}.xml and train_{}.xml already exit (overwrite=False)\n".format(tag,tag))
#         else:
#             ## parameters 
#             if not project_tag:
#                 project_tag = tag
#                 print("No project tag provided - using ml-morph tag.")            
                
#             parameter_checks = {
#                 "project_tag": project_tag,
#                 "mode": mode,
#                 "mask": mask,
#                 "flip": 0,
#                 "split": split,
#                 "prop_train":prop_train,
#                 "prop_test":prop_test,
#                 "n_train":n_train,
#                 "n_test":n_test,
#                 }
            
#             if parameters.__class__.__name__ == "NoneType":
#                 parameters = {}
#             if len(parameters) == 0:
#                 for project_name in self.projects.keys():
#                     parameters[project_name] = {}
#             for project_name in self.projects.keys():
#                 if not project_name in parameters:
#                     parameters[project_name] = {}
#                 for parameter_name, parameter_value in parameter_checks.items():
#                     if not parameter_name in parameters[project_name].keys():
#                         parameters[project_name][parameter_name] = parameter_value
            
#             ## set up xml stuff        
#             train_root = ET.Element('dataset')
#             train_root.append(ET.Element('name'))
#             train_root.append(ET.Element('comment'))
#             train_images_e = ET.Element('images')
#             train_root.append(train_images_e)
            
#             test_global_root = ET.Element('dataset')
#             test_global_root.append(ET.Element('name'))
#             test_global_root.append(ET.Element('comment'))
#             test_global_images_e = ET.Element('images')
#             test_global_root.append(test_global_images_e)
            
#             # =============================================================================
#             # loop through images
            
#             feedback_dict = {}
            
#             for project_name in self.projects.keys():
    
#                 ## fetch project and set up project specific info
#                 parameter = parameters[project_name]
#                 feedback_dict[project_name] = {}
                
#                 ## project specific test-xml files
#                 test_sub_root = ET.Element('dataset')
#                 test_sub_root.append(ET.Element('name'))
#                 test_sub_root.append(ET.Element('comment'))
#                 test_sub_images_e = ET.Element('images')
#                 test_sub_root.append(test_sub_images_e)
                
#                 ## project specific splits
#                 random.seed(random_seed)
#                 proj_dirpaths_shuffled = copy.deepcopy(self.projects[project_name].dir_paths)
#                 random.shuffle(proj_dirpaths_shuffled)
#                 n_total = len(proj_dirpaths_shuffled)
                
#                 val_warning_msg = "WARNING - specified amount of training images equal \
#                       or larger than dataset. You need images for validation!"
#                 test_warning_msg = "No test images specified - using remaining portion"
                            
#                 if parameter["n_train"]:
#                     if parameter["n_train"] >= n_total:
#                         split = n_total
#                         print(val_warning_msg)
#                     else:
#                         split = parameter["n_train"]
#                     if parameter["n_test"]:
#                         end = parameter["n_train"] + parameter["n_test"]
#                     else:
#                         print(test_warning_msg)
#                         end = n_total
#                     if end > n_total:
#                         end = n_total
#                 elif parameter["prop_train"]:
#                     if parameter["prop_train"] == 1:
#                         print(val_warning_msg)
#                     split = int(parameter["prop_train"] * n_total)
#                     if parameter["prop_test"]:
#                         end = split + int(parameter["prop_test"] * n_total)
#                     if end > n_total:
#                         end = n_total
#                 elif parameter["split"]:
#                     split = int(parameter["split"] * n_total)
#                     end = n_total
    
    
#                 for part in ["train","test"]:
            
#                     if part == "train":
#                         start, stop = 0, split
#                     elif part == "test":
#                         start, stop = split, end
        
#                     for idx1, dirpath in enumerate(proj_dirpaths_shuffled[start:stop]):
                        
#                         image = None
                
#                         ## load data
#                         attributes = utils_lowlevel._load_yaml(os.path.join(dirpath, "attributes.yaml"))
#                         annotations = core.export.load_annotation(os.path.join(dirpath, "annotations_" + parameter["project_tag"] + ".json"), verbose=False)
#                         filename = attributes["image_original"]["filename"]
#                         filepath = attributes["image_phenopype"]["filepath"]
#                         image_width, image_height= attributes["image_phenopype"]["width"],  attributes["image_phenopype"]["height"]
    
#                         ## potentially not needed, because img-dirs are on the same level as xml dirs
#                         image_phenopype_path = os.path.abspath(os.path.join(dirpath, attributes["image_phenopype"]["filepath"]))
#                         filepath = os.path.relpath(image_phenopype_path, self.xml_dir)
            
#                         ## feedback
#                         print("Preparing {} data for project {}: {} ({}/{})".format(part, project_name, filename, idx1+1, str(len(proj_dirpaths_shuffled[start:stop]))))       
            
#                         ## checks and feedback
#                         if annotations.__class__.__name__ == "NoneType":
#                             print("No annotations found for {}".format(filename))
#                             continue
#                         if not annotation_type in annotations:
#                             print("No annotation of type {} found for {}".format(annotation_type, filename))
#                             continue
#                         if annotation_id.__class__.__name__ == "NoneType":
#                             annotation_id = max(list(annotations[annotation_type].keys()))
            
#                         ## load landmarks
#                         data = annotations[annotation_type][annotation_id]["data"][annotation_type]
            
#                         ## masking
#                         if parameter["mask"]:
#                             if settings._mask_type in annotations:
#                                 pass
#                             else:
#                                 print("No annotation of type {} found for {}".format(settings._mask_type, filename))
#                                 continue
                        
#                             ## select last mask if no id is given
#                             if mask_id.__class__.__name__ == "NoneType":
#                                 mask_id = max(list(annotations[settings._mask_type].keys()))
            
#                             ## get bounding rectangle and crop image to mask coords
#                             coords = annotations[settings._mask_type][mask_id]["data"][settings._mask_type][0]
#                             rx, ry, rw, rh = cv2.boundingRect(np.asarray(coords, dtype="int32"))
#                         else:
#                             rx, ry, rw, rh = 1, 1, image_width, image_height 
                            
#                         ## flipping
#                         if parameter["flip"]:
                                                    
#                             image = utils.load_image(dirpath)                       
#                             image = cv2.flip(image, 1)
#                             if not rx == 1:
#                                 rx = image_width - (rx + rw)
                                
#                             parameter["mode"] = "save"
                            
#                             data_new = []
#                             for coord in data:
#                                 data_new.append((image_width - coord[0], coord[1]))
#                             data = data_new
                        
#                         ## saving
#                         if parameter["mode"] == "save":
#                             if image.__class__.__name__ == "NoneType":
#                                 image = utils.load_image(dirpath)                       
#                             utils.save_image(image, dir_path=self.image_dir, file_name=filename)
#                             filepath = os.path.relpath(os.path.join(self.image_dir,filename), self.xml_dir)
                            
#                         ## xml part
#                         if part == "train":
#                             train_images_e.append(ml_morph.utils.add_image_element(utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
#                         elif part == "test":
#                             test_global_images_e.append(ml_morph.utils.add_image_element(utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
#                             test_sub_images_e.append(ml_morph.utils.add_image_element(utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
    
#                     ## project specific actions after completing loop
#                     feedback_dict[project_name][part] = len(proj_dirpaths_shuffled[start:stop])
#                     if part == "test":
#                         et = ET.ElementTree(test_sub_root)
#                         xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
#                         with open(os.path.join(self.root_dir,"xml", part + "_" + project_name + "_" + tag + ".xml"), "w") as f:
#                             f.write(xmlstr)
    
#             ## format final XML output
#             for root, part in zip([train_root, test_global_root],["train","test"]):
#                 et = ET.ElementTree(root)
#                 xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
#                 with open(os.path.join(self.root_dir,"xml", part + "_" + tag + ".xml"), "w") as f:
#                     f.write(xmlstr)
                    
#         ## feedback on training data
#         feedback_dict = {}

#         #     train_path = os.path.join(self.xml_dir, "train_{}.xml".format(tag))
#         #     test_path = os.path.join(self.xml_dir, "test_{}.xml".format(tag))
#         #     n_train_imgs = xml_element_counter(train_path, "image")
#         #     n_test_imgs = xml_element_counter(test_path, "image")
#         #     print("Datasets set up for \"{}\":".format(tag))
#         #     print("Total: {} Training data: {} images".format(n_train_imgs))
#         #     print("Test data: {} images".format(n_test_imgs))
#         # else:
            
#         for project_name, project in self.projects.items():
            
#             if len(self.projects) == 1:
#                 test_tag = tag
#             else:
#                 test_tag = project_name + "_" + tag

#             train_path = os.path.join(self.xml_dir, "train_{}.xml".format(tag))
#             test_path = os.path.join(self.xml_dir, "test_{}.xml".format(test_tag))
            
#             n_total = len(project.file_names)
#             n_train_imgs = xml_element_counter(train_path, "image", project)     
#             n_test_imgs = xml_element_counter(test_path, "image")
            
#             print("Prepared datasets for \"{}\" from project \"{}\":".format(tag, project_name))
#             print("total available: {} images - training: {} images - testing: {} images".format(n_total, n_train_imgs, n_test_imgs))
                
                

#     def create_config(
#             self,
#             tag,
#             config_path,
#             overwrite=False,
#             ):

#         if os.path.isfile(config_path):

#             self.config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))

#             if not os.path.isfile(self.config_path):
#                 shutil.copyfile(config_path, self.config_path)
#                 self.model.load_config(self.config_path)
#                 print("- saved a copy at {}".format(self.config_path))
#             else:
#                 self.model.load_config(self.config_path)
#                 print("- found config file at {} - loading (overwrite=False)".format(self.config_path))               
#         else:
#             print("- {} does not exist!".format(config_path))


#     def train_model(self, tag, overwrite=False):

#         ## load config to update recent changes
#         self.model.load_config(self.config_path)

#         print("- training using the following options:\n")
#         config = utils_lowlevel._load_yaml(self.config_path)
#         utils_lowlevel._show_yaml(config["train"])
#         print(" ")
        
#         ## train model
#         self.model.train_model(tag, overwrite)

#     def test_model(self, tag):
        
#         if len(self.projects) == 1:
#             print("Testing global predictor performance:")
#             self.model.test_model(tag)
#         else:
#             for project_name in self.projects.keys():
#                 print("Testing predictor performance on project {}:".format(project_name))
#                 self.model.test_model(tag=tag, test_tag=project_name + "_" + tag)


# def xml_element_counter(xml_path, element_name, project=None):
#     tree, idx = ET.parse(xml_path), 0
#     for elem in tree.iter():
#         if elem.tag == element_name:
#             if project:
#                 if os.path.basename(elem.attrib["file"]) in project.file_names:
#                     pass
#                 else:
#                     continue
#             idx += 1
#     return idx
