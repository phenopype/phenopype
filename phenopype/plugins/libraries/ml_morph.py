#%% imports

clean_namespace = dir()

import copy
import cv2
import numpy as np
import pandas as pd
import os
import random

from dataclasses import make_dataclass

from phenopype import __version__
from phenopype import _config
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

funs = ['attach', 'ProjectWrapper']

def __dir__():
    return clean_namespace + funs


#%% functions

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

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        
        self.model = ml_morph.model.Model(self.root_dir)
        
        if not tag.__class__.__name__ == "NoneType":
            
            active_model_path = os.path.join(self.project.root_dir, "ml_morph", "models", "predictor_{}.dat".format(tag))
            
            if os.path.isfile(active_model_path):
                self.active_model_path = active_model_path
                print("loaded model \"predictor_{}.dat\"".format(tag))
            
    
    def create_training_data(
            self, 
            tag, 
            mode="link",
            overwrite=False, 
            annotation_id=None,
            mask=False,
            mask_id=None,
            random_seed=42,
            split=0.8,
            ):
        
        flags = make_dataclass(
            cls_name="flags", fields=[
                ("overwrite", bool, overwrite),
                ("mode", str, mode),
                ("mask", bool, mask),
                ]
        )
                
        annotation_type = settings._landmark_type
        file_path_save = os.path.join(self.root_dir, "landmarks_ml-morph_" + tag + ".csv")

        if not utils_lowlevel._overwrite_check_file(file_path_save, flags.overwrite):
            return
        
        ## shuffling
        random.seed(random_seed)
        proj_dirpaths_shuffled = copy.deepcopy(self.project.dir_paths)
        random.shuffle(proj_dirpaths_shuffled)
        split = int(split * len(proj_dirpaths_shuffled))

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
                filepath = attributes["image_original"]["filepath"]
                
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
        
    def load_config(self, config_path):
                
        self.model.load_config(config_path)
        
    def train_model(self, tag, overwrite=False):
                
        self.model.train_model(tag, overwrite)
        
    def test_model(self, tag):
                    
        self.model.test_model(tag)
        
    def predict_image(
            self, 
            image, 
            mask=False, 
            tag=None,
            draw=False,
            **kwargs
            ):

        annotations = kwargs.get("annotations", {})
        annotation_type = settings._landmark_type
        annotation_id = kwargs.get("annotation_id", None)

        landmark_tuple_list = []
        
        
        rect = dlib.rectangle(
            left=1, top=1, right=image.shape[1] - 1, bottom=image.shape[0] - 1
        )
        
        if mask:
            if not annotations:
                print("no mask coordinates provided - cannot detect within mask")
                pass
            else:
                annotation_id_mask = kwargs.get(settings._mask_type + "_id", None)
                annotation_mask = utils_lowlevel._get_annotation(
                    annotations,
                    settings._mask_type,
                    annotation_id_mask,
                    prep_msg="- masking regions in thresholded image:",
                )
                
                rx, ry, rw, rh = cv2.boundingRect(np.asarray(annotation_mask["data"][settings._mask_type], dtype="int32"))
                rect = dlib.rectangle(
                    left=rx, top=ry, right=rx+rw, bottom=ry+rh
                )
            

        predictor = dlib.shape_predictor(self.active_model_path)
        predicted_points = predictor(image, rect)
        for item, i in enumerate(sorted(range(0, predicted_points.num_parts), key=str)):
            landmark_tuple_list.append((predicted_points.part(item).x, predicted_points.part(item).y))
            
            
        annotation = {
            "info": {
                "annotation_type": annotation_type,
                "phenopype_function": "plugins.ml_morph.predict_image",
                "phenopype_version": __version__,
            },
            "settings": {
            },
            "data": {
                annotation_type: landmark_tuple_list
                },
        }

        annotations = utils_lowlevel._update_annotations(
            annotations=annotations,
            annotation=annotation,
            annotation_type=annotation_type,
            annotation_id="z",
            kwargs=kwargs,
        )
        
        print(annotations)
        
        if draw:
            canvas = copy.deepcopy(image)
            
            kwargs.pop("annotations",None)
            
            if mask:
                canvas = visualization.draw_mask(canvas, annotations=annotations, **kwargs)
            canvas = visualization.draw_landmark(canvas, annotations=annotations, landmark_id="a", point_colour="red")

            canvas = visualization.draw_landmark(canvas, annotations=annotations, landmark_id="z", point_colour="green")
            utils.show_image(canvas)
            

        return annotations
        