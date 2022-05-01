#%% imports

clean_namespace = dir()

import copy
import cv2
import numpy as np
import pandas as pd
import os
import random
import shutil

from dataclasses import make_dataclass

from phenopype import __version__
from phenopype import _config
from phenopype import main
from phenopype import settings
from phenopype import utils
from phenopype import utils_lowlevel

from phenopype.core import (
    export,
    visualization,
)

import phenomorph as ml_morph
import xml.etree.ElementTree as ET
from xml.dom import minidom
import dlib


#%% namespace cleanup

funs = ['attach', 'Project']

def __dir__():
    return clean_namespace + funs


#%% classes

project = []

def link(project, tag=None):
    if project.__class__.__name__ == "Project":
        return SingleProjectLink(project, tag)
    elif project.__class__.__name__ == "list":
        return MultiProjectLink(project, tag)

class MultiProjectLink:

    def __init__(self, project_list, tag=None):
        
        self.root_dir = os.path.join(os.getcwd(), "ml_morph")

        for project in project_list:
            print(project)


class SingleProjectLink:

    def __init__(self, project, tag=None):

        self.project = project
        self.root_dir = os.path.join(self.project.root_dir, "ml_morph")
        self.image_dir = os.path.join(self.project.root_dir, "ml_morph", "images")

        ## make directories
        self.root_dir = os.path.join(self.root_dir, "ml_morph")
        self.image_dir = os.path.join(self.root_dir, "images")
        self.model_dir = os.path.join(self.root_dir, "models")
        self.config_dir = os.path.join(self.root_dir, "config")
        self.xml_dir = os.path.join(self.root_dir, "xml")

        for directory in [self.root_dir, self.image_dir, self.config_dir, self.xml_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        ## initialize
        self.model = ml_morph.model.Model(self.root_dir)

        ## load existing components
        if not tag.__class__.__name__ == "NoneType":

            model_path = os.path.join(self.model_dir, "predictor_{}.dat".format(tag))
            if os.path.isfile(model_path):
                self.model_path = model_path
                print("- loaded model \"predictor_{}.dat\"".format(tag))

            config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))
            if os.path.isfile(config_path):
                self.config_path = config_path
                print("- loaded config \"config_{}.yaml\"".format(tag))


        ## attach project
        for project_path in project_path_list:
            P = main.Project(project_path)




class _PluginLink(object):
    def __init__(
            self,
            phenopype_project,
            tag=None
            ):
        """
        Internal function to attach the ml-morph routine to a phenopype project.
        Not to be used by itself.

        Parameters
        ----------
        phenopype_project : Project
            a single phenopype project.
        tag : str, optional
            a tag to be used for the ml-morph procedure

        Returns
        -------
        None.

        """

        ## attach project
        self.phenopype_project = phenopype_project

        ## make directories
        self.root_dir = os.path.join(self.phenopype_project.root_dir, "ml_morph")
        self.image_dir = os.path.join(self.root_dir, "images")
        self.model_dir = os.path.join(self.root_dir, "models")
        self.config_dir = os.path.join(self.root_dir, "config")
        self.xml_dir = os.path.join(self.root_dir, "xml")

        for directory in [self.root_dir, self.image_dir, self.config_dir, self.xml_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        ## initialize
        self.model = ml_morph.model.Model(self.root_dir)

        ## load existing components
        if not tag.__class__.__name__ == "NoneType":

            model_path = os.path.join(self.model_dir, "predictor_{}.dat".format(tag))
            if os.path.isfile(model_path):
                self.model_path = model_path
                print("- loaded model \"predictor_{}.dat\"".format(tag))

            config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))
            if os.path.isfile(config_path):
                self.config_path = config_path
                print("- loaded config \"config_{}.yaml\"".format(tag))

    def create_training_data(
            self,
            tag,
            mode="link",
            overwrite=False,
            annotation_id=None,
            mask=False,
            mask_id=None,
            rotate=0,
            random_seed=42,
            split=0.8,
            ):

        # =============================================================================
        # setup

        ## define flags
        flags = make_dataclass(
            cls_name="flags", fields=[
                ("overwrite", bool, overwrite),
                ("mode", str, mode),
                ("mask", bool, mask),
                ("rotate", int, 0),
                ]
        )

        annotation_type = settings._landmark_type

        ## overwrite check
        ret = utils_lowlevel._file_walker(self.xml_dir, include=[tag])[0]
        if len(ret) > 0 and not flags.overwrite:
            print("test_{}.xml and train_{}.xml already exit (overwrite=False)".format(tag,tag))
            return

        ## shuffling
        random.seed(random_seed)
        proj_dirpaths_shuffled = copy.deepcopy(self.phenopype_project.dir_paths)
        random.shuffle(proj_dirpaths_shuffled)
        split = int(split * len(proj_dirpaths_shuffled))

        # =============================================================================
        # loop through images


        for part in ["train","test"]:

            ## XML stuff
            root = ET.Element('dataset')
            root.append(ET.Element('name'))
            root.append(ET.Element('comment'))
            images_e = ET.Element('images')
            root.append(images_e)

            if part == "train":
                start, stop = 0, split
            elif part == "test":
                start, stop = split, len(proj_dirpaths_shuffled)

            for idx1, dirpath in enumerate(proj_dirpaths_shuffled[start:stop], 1):

                ## load data
                attributes = utils_lowlevel._load_yaml(os.path.join(dirpath, "attributes.yaml"))
                annotations = export.load_annotation(os.path.join(dirpath, "annotations_" + tag + ".json"), verbose=False)
                filename = attributes["image_original"]["filename"]
                filepath = attributes["image_phenopype"]["filepath"]

                print(filepath)

                # ## potentially not needed, because img-dirs are on the same level as xml dirs
                # image_phenopype_path = os.path.abspath(os.path.join(dirpath, attributes["image_phenopype"]["filepath"]))
                # filepath = os.path.relpath(image_phenopype_path, self.xml_dir)


                print("Preparing {} data for: ({}/{}) ".format(part, idx1, len(proj_dirpaths_shuffled[start:stop])) + filename)

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

                ## load image if masking or resizing
                if flags.mask and settings._mask_type in annotations:

                    ## select last mask if no id is given
                    if mask_id.__class__.__name__ == "NoneType":
                        mask_id = max(list(annotations[settings._mask_type].keys()))

                    ## get bounding rectangle and crop image to mask coords
                    coords = annotations[settings._mask_type][mask_id]["data"][settings._mask_type][0]
                    rx, ry, rw, rh = cv2.boundingRect(np.asarray(coords, dtype="int32"))

                ## xml part
                images_e.append(ml_morph.utils.add_image_element(utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))

            ## format final XML output
            et = ET.ElementTree(root)
            xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
            with open(os.path.join(self.root_dir,"xml", part + "_" + tag + ".xml"), "w") as f:
                f.write(xmlstr)

    def create_config(
            self,
            tag,
            config_path,
            ):


        if os.path.isfile(config_path):

            self.config_path = os.path.join(self.config_dir, "config_{}.yaml".format(tag))

            if not os.path.isfile(self.config_path):
                shutil.copyfile(config_path, self.config_path)
                self.model.load_config(self.config_path)
                print("- saved a copy at {}".format(self.config_path))
            else:
                self.model.load_config(self.config_path)
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
