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

funs = ['ProjectBinder']

def __dir__():
    return clean_namespace + funs


#%% classes

project = []


class ProjectLink:

    def __init__(
            self, 
            projects, 
            root_dir,
            tag=None,
            load=True,
            overwrite=False,
            **kwargs,
            ):
        
        # =============================================================================
        # setup
        
        ## create flags
        flags = make_dataclass(
            cls_name="flags", fields=[
                ("overwrite", bool, overwrite),
                ("load", bool, load),
                ]
        )
        
        ## list check and attach projects
        self.projects = {}
        if not projects.__class__.__name__ == "list":
            projects = [projects]
                
        for project in projects:
            if project.__class__.__name__ == "str":
                if os.path.isdir(project):
                    project = main.Project(project)
            project_name = os.path.basename(project.root_dir)    
            self.projects[project_name] = project

        ## multi-check and make root dir
        if len(self.projects) > 1:
            self.root_dir = copy.deepcopy(root_dir)
        else:
            self.root_dir = copy.deepcopy(os.path.join(root_dir, "ml_morph"))       
                        
        ## make other directories
        self.image_dir = os.path.join(self.root_dir, "images")
        self.model_dir = os.path.join(self.root_dir, "models")
        self.config_dir = os.path.join(self.root_dir, "config")
        self.xml_dir = os.path.join(self.root_dir, "xml")
        for directory in [self.root_dir, self.image_dir, self.config_dir, self.xml_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        ## attach model
        self.model = ml_morph.model.Model(self.root_dir)
        if len(self.projects) > 1:            
            print("- {} projects loaded ({})".format(len(self.projects), ', '.join(list(self.projects.keys()))))

        ## load existing components
        if not tag.__class__.__name__ == "NoneType":
            ret = utils_lowlevel._file_walker(self.xml_dir, include=[tag], pype_mode=True)[0]
            if len(ret) > 0:
                print("- found training and test datasets \"test_{}.xml\" and \"train_{}.xml\"".format(tag, tag))
            config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))
            if os.path.isfile(config_path):
                self.config_path = config_path
                print("- loaded config \"config_{}.yaml\"".format(tag))
            model_path = os.path.join(self.model_dir, "predictor_{}.dat".format(tag))
            if os.path.isfile(model_path):
                self.model_path = model_path
                print("- loaded model \"predictor_{}.dat\"".format(tag))

    def create_training_data(
            self,
            tag,
            mode="link",
            overwrite=False,
            landmark_id=None,
            mask=False,
            mask_id=None,
            flip=False,
            random_seed=42,
            split=0.8,
            parameters=None,
            ):
    
        # =============================================================================
        # setup
    
        ## define flags
        flags = make_dataclass(
            cls_name="flags", fields=[
                ("overwrite", bool, overwrite),
                ]
        )
    
        annotation_type = settings._landmark_type
        annotation_id = landmark_id
        
        ## overwrite check
        ret = utils_lowlevel._file_walker(self.xml_dir, include=[tag], pype_mode=True)[0]
        if len(ret) > 0 and not flags.overwrite:
            print("test_{}.xml and train_{}.xml already exit (overwrite=False)".format(tag,tag))
            return
        else:
            pass
                        
        ## parameter checks
        parameter_checks = {
            "mode": mode,
            "mask": mask,
            "flip": 0,
            "split": split,
            }
        
        if parameters.__class__.__name__ == "NoneType":
            parameters = {}
        if len(parameters) == 0:
            for project_name in self.projects.keys():
                parameters[project_name] = {}
        for project_name in self.projects.keys():
            if not project_name in parameters:
                parameters[project_name] = {}
            for parameter_name, parameter_value in parameter_checks.items():
                if not parameter_name in parameters[project_name].keys():
                    parameters[project_name][parameter_name] = parameter_value
        
        ## set up xml stuff        
        train_root = ET.Element('dataset')
        train_root.append(ET.Element('name'))
        train_root.append(ET.Element('comment'))
        train_images_e = ET.Element('images')
        train_root.append(train_images_e)
        
        test_root = ET.Element('dataset')
        test_root.append(ET.Element('name'))
        test_root.append(ET.Element('comment'))
        test_images_e = ET.Element('images')
        test_root.append(test_images_e)
        
        # =============================================================================
        # loop through images
        
        feedback_dict = {}
        
        for project_name in self.projects.keys():

            parameter = parameters[project_name]
            feedback_dict[project_name] = {}
            
            ## project specific splits
            proj_dirpaths_shuffled = copy.deepcopy(self.projects[project_name].dir_paths)
            random.shuffle(proj_dirpaths_shuffled)
            split = int(parameter["split"] * len(proj_dirpaths_shuffled))
    
            for part in ["train","test"]:
        
                if part == "train":
                    start, stop = 0, split
                elif part == "test":
                    start, stop = split, len(proj_dirpaths_shuffled)
    
                for idx1, dirpath in enumerate(proj_dirpaths_shuffled[start:stop], 1):
            
                    ## load data
                    attributes = utils_lowlevel._load_yaml(os.path.join(dirpath, "attributes.yaml"))
                    annotations = core.export.load_annotation(os.path.join(dirpath, "annotations_" + tag + ".json"), verbose=False)
                    filename = attributes["image_original"]["filename"]
                    filepath = attributes["image_phenopype"]["filepath"]
                    image_width, image_height= attributes["image_phenopype"]["width"],  attributes["image_phenopype"]["height"]

                    ## potentially not needed, because img-dirs are on the same level as xml dirs
                    image_phenopype_path = os.path.abspath(os.path.join(dirpath, attributes["image_phenopype"]["filepath"]))
                    filepath = os.path.relpath(image_phenopype_path, self.xml_dir)
        
                    ## feedback
                    print("Preparing {} data for project {}: {} ({}/{})".format(part, project_name, filename, idx1, str(len(proj_dirpaths_shuffled[start:stop]))))       
        
                    ## checks and feedback
                    if annotations.__class__.__name__ == "NoneType":
                        print("No annotations found for {}".format(filename))
                        continue
                    if not annotation_type in annotations:
                        print("No annotation of type {} found for {}".format(annotation_type, filename))
                        continue
                    if annotation_id.__class__.__name__ == "NoneType":
                        annotation_id = max(list(annotations[annotation_type].keys()))
        
                    ## load landmarks
                    data = annotations[annotation_type][annotation_id]["data"][annotation_type]
        
                    ## masking
                    if parameter["mask"]:
                        if settings._mask_type in annotations:
                            pass
                        else:
                            print("No annotation of type {} found for {}".format(settings._mask_type, filename))
                            continue
                    
                        ## select last mask if no id is given
                        if mask_id.__class__.__name__ == "NoneType":
                            mask_id = max(list(annotations[settings._mask_type].keys()))
        
                        ## get bounding rectangle and crop image to mask coords
                        coords = annotations[settings._mask_type][mask_id]["data"][settings._mask_type][0]
                        rx, ry, rw, rh = cv2.boundingRect(np.asarray(coords, dtype="int32"))
                    else:
                        rx, ry, rw, rh = 1, 1, image_width, image_height 
                        
                    ## flipping
                    if parameter["flip"]:
                        
                        image = utils.load_image(dirpath)                       
                        image = cv2.flip(image, 1)
                        if not rx == 1:
                            rx = image_width - (rx + rw)
                            
                        parameter["mode"] = "save"
                        
                        data_new = []
                        for coord in data:
                            data_new.append((image_width - coord[0], coord[1]))
                        data = data_new
                    
                    ## saving
                    if parameter["mode"] == "save":
                        utils.save_image(image, dir_path=self.image_dir, file_name=filename)
                        filepath = os.path.relpath(os.path.join(self.image_dir,filename), self.xml_dir)
                        
                    ## xml part
                    if part == "train":
                        train_images_e.append(ml_morph.utils.add_image_element(utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
                    elif part == "test":
                        test_images_e.append(ml_morph.utils.add_image_element(utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
                        
                feedback_dict[project_name][part] = len(proj_dirpaths_shuffled[start:stop])

        ## format final XML output
        for root, part in zip([train_root, test_root],["train","test"]):
            et = ET.ElementTree(root)
            xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
            with open(os.path.join(self.root_dir,"xml", part + "_" + tag + ".xml"), "w") as f:
                f.write(xmlstr)
                
        print("\nTraining dataset created:\n")
        print(json.dumps(feedback_dict))

    def create_config(
            self,
            tag,
            config_path,
            overwrite=False,
            ):

        if os.path.isfile(config_path):

            self.config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))

            if not os.path.isfile(self.config_path):
                shutil.copyfile(config_path, self.config_path)
                self.model.load_config(self.config_path)
                print("- saved a copy at {}".format(self.config_path))
            else:
                self.model.load_config(self.config_path)
                print("- found config file at {} - loading (overwrite=False)".format(self.config_path))               
        else:
            print("- {} does not exist!".format(config_path))


    def train_model(self, tag, overwrite=False):

        ## load config to update recent changes
        self.model.load_config(self.config_path)

        print("- training using the following options:\n")
        config = utils_lowlevel._load_yaml(self.config_path)
        utils_lowlevel._show_yaml(config["train"])
        print(" ")
        
        ## train model
        self.model.train_model(tag, overwrite)

    def test_model(self, tag):

        self.model.test_model(tag)




# class _ProjectLink(object):
#     def __init__(
#             self,
#             phenopype_project,
#             tag=None,
#             ):
#         """
#         Internal function to attach the ml-morph routine to a phenopype project.
#         Not to be used by itself.

#         Parameters
#         ----------
#         phenopype_project : Project
#             a single phenopype project.
#         tag : str, optional
#             a tag to be used for the ml-morph procedure

#         Returns
#         -------
#         None.

#         """

#         ## attach project
#         self.phenopype_project = phenopype_project

#         ## make directories
#         self.root_dir = os.path.join(self.phenopype_project.root_dir, "ml_morph")
#         self.image_dir = os.path.join(self.root_dir, "images")
#         self.model_dir = os.path.join(self.root_dir, "models")
#         self.config_dir = os.path.join(self.root_dir, "config")
#         self.xml_dir = os.path.join(self.root_dir, "xml")

#         for directory in [self.root_dir, self.image_dir, self.config_dir, self.xml_dir]:
#             if not os.path.exists(directory):
#                 os.makedirs(directory)

#         ## initialize
#         self.model = ml_morph.model.Model(self.root_dir)

#         ## load existing components
#         if not tag.__class__.__name__ == "NoneType":

#             model_path = os.path.join(self.model_dir, "predictor_{}.dat".format(tag))
#             if os.path.isfile(model_path):
#                 self.model_path = model_path
#                 print("- loaded model \"predictor_{}.dat\"".format(tag))

#             config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))
#             if os.path.isfile(config_path):
#                 self.config_path = config_path
#                 print("- loaded config \"config_{}.yaml\"".format(tag))
 
#     def create_training_data(
#             self,
#             tag,
#             mode="link",
#             overwrite=False,
#             annotation_id=None,
#             mask=False,
#             mask_id=None,
#             rotate=0,
#             random_seed=42,
#             split=0.8,
#             ):

#         # =============================================================================
#         # setup

#         ## define flags
#         flags = make_dataclass(
#             cls_name="flags", fields=[
#                 ("overwrite", bool, overwrite),
#                 ("mode", str, mode),
#                 ("mask", bool, mask),
#                 ("rotate", int, 0),
#                 ]
#         )

#         annotation_type = settings._landmark_type

#         ## overwrite check
#         ret = utils_lowlevel._file_walker(self.xml_dir, include=[tag])[0]
#         if len(ret) > 0 and not flags.overwrite:
#             print("test_{}.xml and train_{}.xml already exit (overwrite=False)".format(tag,tag))
#             return

#         ## shuffling
#         random.seed(random_seed)
#         proj_dirpaths_shuffled = copy.deepcopy(self.phenopype_project.dir_paths)
#         random.shuffle(proj_dirpaths_shuffled)
#         split = int(split * len(proj_dirpaths_shuffled))

#         # =============================================================================
#         # loop through images


#         for part in ["train","test"]:

#             ## XML stuff
#             root = ET.Element('dataset')
#             root.append(ET.Element('name'))
#             root.append(ET.Element('comment'))
#             images_e = ET.Element('images')
#             root.append(images_e)

#             if part == "train":
#                 start, stop = 0, split
#             elif part == "test":
#                 start, stop = split, len(proj_dirpaths_shuffled)

#             for idx1, dirpath in enumerate(proj_dirpaths_shuffled[start:stop], 1):

#                 ## load data
#                 attributes = utils_lowlevel._load_yaml(os.path.join(dirpath, "attributes.yaml"))
#                 annotations = export.load_annotation(os.path.join(dirpath, "annotations_" + tag + ".json"), verbose=False)
#                 filename = attributes["image_original"]["filename"]
#                 filepath = attributes["image_phenopype"]["filepath"]

#                 # ## potentially not needed, because img-dirs are on the same level as xml dirs
#                 # image_phenopype_path = os.path.abspath(os.path.join(dirpath, attributes["image_phenopype"]["filepath"]))
#                 # filepath = os.path.relpath(image_phenopype_path, self.xml_dir)

#                 print("Preparing {} data for: ({}/{}) ".format(part, idx1, len(proj_dirpaths_shuffled[start:stop])) + filename)

#                 ## checks and feedback
#                 if annotations.__class__.__name__ == "NoneType":
#                     print("No annotations found for {}".format(filename))
#                     continue
#                 if not annotation_type in annotations:
#                     print("No annotation of type {} found for {}".format(annotation_type, filename))
#                     continue
#                 if annotation_id.__class__.__name__ == "NoneType":
#                     annotation_id = max(list(annotations[annotation_type].keys()))

#                 ## load landmarks
#                 data = annotations[annotation_type][annotation_id]["data"][annotation_type]

#                 ## load image if masking or resizing
#                 if flags.mask and settings._mask_type in annotations:

#                     ## select last mask if no id is given
#                     if mask_id.__class__.__name__ == "NoneType":
#                         mask_id = max(list(annotations[settings._mask_type].keys()))

#                     ## get bounding rectangle and crop image to mask coords
#                     coords = annotations[settings._mask_type][mask_id]["data"][settings._mask_type][0]
#                     rx, ry, rw, rh = cv2.boundingRect(np.asarray(coords, dtype="int32"))

#                 ## xml part
#                 images_e.append(ml_morph.utils.add_image_element(utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))

#             ## format final XML output
#             et = ET.ElementTree(root)
#             xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
#             with open(os.path.join(self.root_dir,"xml", part + "_" + tag + ".xml"), "w") as f:
#                 f.write(xmlstr)

#     def create_config(
#             self,
#             tag,
#             config_path,
#             ):


#         if os.path.isfile(config_path):

#             self.config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))

#             if not os.path.isfile(self.config_path):
#                 shutil.copyfile(config_path, self.config_path)
#                 self.model.load_config(self.config_path)
#                 print("- saved a copy at {}".format(self.config_path))
#             else:
#                 self.model.load_config(self.config_path)
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

#         self.model.test_model(tag)
