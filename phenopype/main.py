#%% imports

import copy
import json
import os
import numpy as np
import pandas as pd
import platform
import string
import io 
import sys
import logging

from contextlib import redirect_stdout
from dataclasses import make_dataclass

import pprint
import subprocess
import time
import zipfile

import cv2
import ruamel.yaml
from datetime import datetime

import shutil
from ruamel.yaml.comments import CommentedMap as ordereddict
from collections import defaultdict, deque

from phenopype import __version__
from phenopype import _config
from phenopype import settings
from phenopype import utils
from phenopype import utils_lowlevel as ul

from phenopype.core import (
    preprocessing,
    segmentation,
    measurement,
    export,
    visualization,
)

#%% settings

pd.options.display.max_rows = (
    settings.pandas_max_rows
)  # how many rows of pd-dataframe to show
pretty = pprint.PrettyPrinter(width=30)  # pretty print short strings
ruamel.yaml.Representer.add_representer(
    ordereddict, ruamel.yaml.Representer.represent_dict
)  # suppress !!omap node info

#%% classes

class Project_labelling:
    
    def __init__(
            self, 
            root_dir, 
            load=True, 
            check=False,
            overwrite=False,
            ask=True,
            **kwargs,
            ):
        """
        

        Parameters
        ----------
        root_dir : TYPE
            DESCRIPTION.
        load : TYPE, optional
            DESCRIPTION. The default is True.
        check : TYPE, optional
            DESCRIPTION. The default is False.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        ask : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ## set flags
        flags = make_dataclass(
            cls_name="flags",
            fields=[("load", bool, load), 
                    ("overwrite", bool, overwrite),
                    ("check", bool, check),
                    ("check_path", "str", kwargs.get("check_path","abs")),
                    ("ask", bool, ask),

                    ],
        )

        ## path conversion
        root_dir = root_dir.replace(os.sep, "/")
        root_dir = os.path.abspath(root_dir)

        print("--------------------------------------------")
        while True:
            if os.path.isdir(root_dir):
                if all(
                    [
                        "attributes.yaml" in os.listdir(root_dir),
                        "data" in os.listdir(root_dir),
                    ]
                ):
                    if flags.load and not flags.overwrite:
                        print(
                            "Found existing project root directory - loading from:\n"
                            + root_dir
                        )
                        break
                    elif not flags.load and flags.overwrite:
                        pass
                    elif flags.load and flags.overwrite:
                        print(
                            "Found existing phenopype project directory at:\n{}\n".format(
                                root_dir
                            )
                        )
                        time.sleep(1)
                        query1 = input("overwrite (y/n)?")
                        if query1 in settings.confirm_options:
                            pass
                        else:
                            print(
                                'Aborted - project "{}" not overwritten'.format(
                                    os.path.basename(root_dir)
                                )
                            )
                            return
                    shutil.rmtree(root_dir, onerror=ul._del_rw)
                    os.makedirs(root_dir)
                    os.makedirs(os.path.join(root_dir, "data"))
                    print('\n"' + root_dir + '" created (overwritten)')
                    break
                else:
                    print(
                        "Directory is neither empty nor a valid phenopype directory - aborting."
                    )
                    return
            else:
                print(
                    "Creating a new phenopype project directory at:\n" + root_dir + "\n"
                )
                if flags.ask:
                    query2 = input("Proceed? (y/n)\n")
                else:
                    query2 = "y"
                if query2 in settings.confirm_options:
                    os.makedirs(root_dir)
                    os.makedirs(os.path.join(root_dir, "data"))
                    break
                else:
                    print('\n"' + root_dir + '" not created!')
                    return

        ## global project attributes
        project_attributes_path = os.path.join(root_dir, "attributes.yaml")
        if not os.path.isfile(project_attributes_path):
            project_attributes = {
                "project_info": {
                    "date_created": datetime.today().strftime(settings.strftime_format),
                    "date_changed": datetime.today().strftime(settings.strftime_format),
                    "phenopype_version": __version__,
                },
                "project_data": {
                    "source": {},
                    "progress": {},
                    },
            }
            ul._save_yaml(
                project_attributes, os.path.join(root_dir, "attributes.yaml")
            )
            print(
                '\nProject "{}" successfully created.'.format(
                    os.path.basename(root_dir)
                )
            )
        else:
            project_attributes = ul._load_yaml(project_attributes_path, typ="safe")
            

        print("--------------------------------------------")
        
        ## attach to instance
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.attributes = project_attributes

        if "images.json" in os.listdir(self.data_dir):
            with open(os.path.join(self.data_dir, "images.json")) as file:
                self.file_dict = json.load(file)
            print('\nLabelling project "{}" successfully loaded with {} images'.format(
                    os.path.basename(root_dir), len(self.file_dict)
                ))
        else:
            self.file_dict = {}
            if not flags.overwrite:
                print('\nLablling project "{}" successfully loaded, but it didn\'t contain any images!'.format(
                        os.path.basename(root_dir)
                    ))

        if flags.check:
            print("Checking for missing files:")
            missing = []
            for key, value in self.file_dict.items():
                if flags.check_path == "abs":
                    if not os.path.isfile(value["filepath_abs"]):
                        missing.append(value["filepath_abs"])
                if flags.check_path == "rel":
                    if not os.path.isfile(value["filepath_rel"]):
                        missing.append(value["filepath_rel"])
            if len(missing) > 0:
                self.missing = missing
                print(f"- found {len(missing)} images (access with project.missing)")
            elif len(missing) == 0:
                print("- all files found!")   
    
    def add_files(
        self,
        images,
        filetypes=settings.default_filetypes,
        include=[],
        include_all=True,
        exclude=[],
        n_max=None,
        recursive=False,
        overwrite=False,
        unique="path",
        **kwargs
    ):
        """
        Add files to the labeling tool.

        Parameters
        ----------
        images : str or list
            Either a directory containing images or a list of image paths.
        filetypes : list, optional
            List of file extensions to include. Default is settings.default_filetypes.
        include : list, optional
            List of patterns to include. Default is [].
        include_all : bool, optional
            Whether to include all patterns in include. Default is True.
        exclude : list, optional
            List of patterns to exclude. Default is [].
        n_max : int, optional
            Maximum number of files to add. Default is None (add all files).
        recursive : bool, optional
            Whether to search directories recursively. Default is False.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        unique : str, optional
            Strategy to handle duplicate files. Default is "path".
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None

        """

        # Set up
        indent = kwargs.get("indent", 4)
        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("recursive", bool, recursive),
                ("overwrite", bool, overwrite),
            ],
        )

        if isinstance(images, list):
            filepaths = images
            image_dir = "from-list"
        elif isinstance(images, str):
            image_dir = os.path.abspath(images.replace(os.sep, "/"))

            # Feedback
            print("--------------------------------------------")
            print(f"phenopype will search for image files at\n{image_dir}\n")
            print("Using the following settings:\n"
                  f"filetypes: {filetypes}, include: {include}, exclude: {exclude}, "
                  f"recursive: {flags.recursive}, unique: {unique}\n")

            # Collect filepaths if not already provided
            if not isinstance(images, list):
                filepaths, _ = ul._file_walker(
                    directory=image_dir,
                    recursive=recursive,
                    unique=unique,
                    filetypes=filetypes,
                    exclude=exclude,
                    include=include,
                    include_all=include_all,
                )

        # Subsetting
        n_total_found = len(filepaths)
        n_max = str(n_max) if n_max is not None else "all"
        filepaths = filepaths[:n_max] if n_max != "all" else filepaths

        # Loop through files
        for filepath in filepaths:
            # Image name and extension
            image_name = os.path.basename(filepath)

            if image_name in self.file_dict:
                if not flags.overwrite:
                    print(f"Image {image_name} already exists (overwrite=False).")
                    continue
                else:
                    print(f"Image {image_name} already exists - overwriting!")

            else:
                print(f"Image {image_name} found - adding to list.")

            self.file_dict[image_name] = {
                "filepath_abs": filepath,
                "filepath_rel": os.path.relpath(filepath, self.root_dir),
            }

        # Add dirlists to project object (always overwrite)
        self.filenames = list(self.file_dict.keys())

        print(f"\nFound {len(self.filenames)} files - using {n_max}")
        print("--------------------------------------------")

        # Check existing files
        images_json_path = os.path.join(self.data_dir, "images.json")
        if os.path.exists(images_json_path):
            with open(images_json_path, "r") as file:
                file_dict = json.load(file)
        else:
            file_dict = {}

        # If new files
        if self.file_dict != file_dict:
            # Update image-data
            with open(images_json_path, "w") as file:
                json.dump(self.file_dict, file, indent=indent)
                print("- saved image.json.")

            # Update project attributes
            self.attributes["project_data"]["source"][image_dir] = {
                "found images": n_total_found,
                "using": n_max,
                "added/modified": datetime.today().strftime(settings.strftime_format),
            }
            self.attributes["project_info"]["date_changed"] = datetime.today().strftime(
                settings.strftime_format)
            ul._save_yaml(self.attributes, os.path.join(self.root_dir, "attributes.yaml"))
        else:
            print("- no new files - nothing to save.")
            
 
    def run(
        self,
        tag,
        config_path,
        overwrite=False,
        image_path="abs",
        **kwargs,
        ):
        """
        

        Parameters
        ----------
        tag : TYPE
            DESCRIPTION.
        config_path : TYPE
            DESCRIPTION.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        image_path : TYPE, optional
            DESCRIPTION. The default is "abs".
        **kwargs : TYPE
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # =============================================================================
        # setup
        
        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("overwrite", bool, overwrite),
                ("image_path", str, image_path),
                ],
        )
        
        ## attributes
        self.tag = tag
        self.config = ul._load_yaml(config_path, typ="safe")       
        self.labels_filepath = os.path.join(self.data_dir, f"{tag}_labels.json")
        self.image_list = list(self.file_dict)
        self.kwargs = kwargs
        self.current = make_dataclass(cls_name="vars", fields=[])     
        self.current.exit = False
        self.current.idx = 0

        ## load labels
        if os.path.isfile(self.labels_filepath):
            with open(self.labels_filepath, "r") as file:
                self.labels = json.load(file)
            for img_name, label in self.labels.items():
                if "mask" in label:
                    if "coords" in label["mask"]:
                        label["mask"]["coords"] = ul._NoIndent(label["mask"]["coords"])
                        self.labels[img_name] = label
        else:
            self.labels = {}
            
        ## continue
        if self.tag in self.attributes["project_data"]["progress"]:
            self.current.idx = self.attributes["project_data"]["progress"][self.tag]["current_idx"]
            self.current.image_name = self.attributes["project_data"]["progress"][self.tag]["current_image"]
            
        # assert self.current.image_name == self.image_list[self.current.idx]
        
        # ============================================================================
        # run

        ## keeps pumping images unless ended with esc 
        while not self.current.exit:
                                                            
            ## navigation
            self.current.idx_prev = copy.deepcopy(self.current.idx)
            self.current.image_name = self.image_list[self.current.idx]
            self.current.image_info = self.file_dict[self.current.image_name]
            self.current.filepath = self.current.image_info["filepath_" + flags.image_path]
            self.current.image = utils.load_image(self.current.filepath)         
            self.current.image_folder = os.path.basename(os.path.dirname(self.current.filepath))
            
            ## fetch label
            if self.current.image_name in self.labels:
                self.label = self.labels[self.current.image_name] 
            else:
                self.label = defaultdict(dict)
            
            ## go through config
            for idx, (step_name, step) in enumerate(self.config.items()):
                if step_name == "text":
                    brk = self._text(self.current.image)
                if step_name == "mask":
                    brk = self._mask(self.current.image)
                if brk:
                    break
                
            ## ensure to advance at end of config
            if self.current.idx == self.current.idx_prev:
                if self.current.idx == 0 and self.current.keypress == 2424832:
                    print("Beginning of list!")
                elif self.current.idx == len(self.image_list) and self.current.keypress == 2555904:
                    print("End of list!")
                else:
                    self.current.idx += 1
                     
    def export(
        self,
        tag,
        category_column=False,
        overwrite=False,
        save_dir=None,
        **kwargs,
        ):
        
        """
        Export labels DataFrame to a specified directory as a CSV file.

        Parameters
        ----------
        tag : str
            A tag for the export operation.
        category_column : bool, optional
            Whether to use the 'category' column or make it its own column. Default is False.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        save_dir : str, optional
            Directory to save the CSV file. Defaults to self.root_dir if not provided.
        **kwargs : dict
            Additional keyword arguments. Unused in this method.

        Returns
        -------
        None
        """

        # Set up
        self.tag = tag
        if save_dir is None:
            save_dir = os.path.join(self.root_dir, "export")
            os.makedirs(save_dir, exist_ok=True)
        self.labels_filepath = os.path.join(self.root_dir, f"{tag}_labels.json")
        self.image_list = list(self.file_dict)

        # Load labels
        if os.path.isfile(self.labels_filepath):
            with open(self.labels_filepath, "r") as file:
                self.labels = json.load(file)
            for img_name, label in self.labels.items():
                if "mask" in label:
                    if "coords" in label["mask"]:
                        label["mask"]["coords"] = ul._NoIndent(label["mask"]["coords"])
                        self.labels[img_name] = label

        # Convert labels to DataFrame
        flat_dict = {}
        for key, value in self.labels.items():
            category = value['text']['category']
            label = value['text']['label']
            flat_dict[key] = {'category': category, 'label': label}

        self.labels_df = pd.DataFrame.from_dict(flat_dict, orient='index')
        
        # Handle category_column option
        if category_column:
            self.labels_df = self.labels_df.reset_index().rename(
                columns={'index': 'filename', 'category': 'category', 'label': 'label'})
        else:
            self.labels_df = self.labels_df.reset_index().rename(
                columns={'index': 'filename'}).pivot(index='filename', columns='category', values='label')

        # Save DataFrame to CSV
        save_path = os.path.join(save_dir, f"{tag}_labels.csv")
        if os.path.exists(save_path) and not overwrite:
            raise FileExistsError("File already exists. Set overwrite=True to overwrite.")
        else:
            self.labels_df.to_csv(save_path)
            print(f"labels saved to: {save_path}")
        
        
    def _keypress_eval(self):

        if self.current.keypress == 27:
            cv2.destroyAllWindows()
            self._save()
            self.current.exit = True
            return True
        elif self.current.keypress == 2424832:
            self.current.idx -= 1
            self.current.idx = max(self.current.idx, 0)
            return True
        elif self.current.keypress == 2555904:
            self.current.idx += 1
            self.current.idx = min(self.current.idx, len(self.image_list))
            return True
        elif self.current.keypress == 13:
            return False
        else:
            self.current.idx += 1
            return False
            
    def _text(self, image, **kwargs):
                
        ## check for existing label 
        try:   
            cat, lab = self.label["text"]["category"], self.label["text"]["label"]
        except:   
            self.label["text"] = {}
            cat, lab = self.config["text"]["category"], ""

        ## check for args in options and fun call
        if "options" in self.config["text"]:
            kwargs.update(self.config["text"]["options"])
        kwargs.update(self.kwargs)
            
        gui = ul._GUI(
            image,
            window_aspect="normal",
            window_name="labelling-tool",
            tool="labelling",
            labelling=True,
            query=cat,
            label_keymap=self.config["text"]["keymap"],
            data={settings._comment_type: lab},
            **kwargs,
        )
        
        ## save label
        lab = gui.data[settings._comment_type]
        self.label["text"]["category"] = cat
        self.label["text"]["label"] = lab
        self.labels[self.current.image_name] = self.label     
    
        ## navigation
        self.current.keypress = gui.keypress 
        return self._keypress_eval()


    def _mask(self,image, **kwargs):
        
        ## check for existing label 
        try:   
            cat, coords = self.label["mask"]["category"], self.label["mask"]["coords"]
            if not type(coords) == list:
                coords = coords.to_list()
        except:   
            self.label["mask"] = {}
            cat, coords = self.config["mask"]["category"], []
            
        ## check for args in options and fun call        
        if "options" in self.config["text"]:
            kwargs.update(self.config["mask"]["options"])
        kwargs.update(self.kwargs)

        gui = ul._GUI(
            image,
            window_aspect="normal",
            window_name="labelling-tool",
            labelling=True,
            data={settings._coord_list_type: coords},
            **kwargs,
        )
         
        ## save label
        self.label["mask"]["category"] = cat
        self.label["mask"]["coords"] = ul._NoIndent(gui.data["polygons"])
        self.labels[self.current.image_name] = self.label     
        
        ## navigation
        self.current.keypress = gui.keypress 
        return self._keypress_eval()
        
    def _comment(
        self,
        image):
        
        pass 
    
    def _save(self):
        
        print("Exiting:")
                  
        ## save json
        if os.path.exists(self.labels_filepath):
            
            labels_filepath_temp = self.labels_filepath + '.temp'
            with open(labels_filepath_temp, 'w') as temp_file:
                json.dump(self.labels, temp_file, indent=4, cls=ul._NoIndentEncoder)
        
            # Create up to three backups of the original file
            backup_dir = os.path.dirname(self.labels_filepath)
            backup_prefix = os.path.basename(self.labels_filepath) + '_backup_'
            backups = deque(sorted(filter(lambda x: x.startswith(backup_prefix), os.listdir(backup_dir))))
        
            # Create a backup of the original file with the current datetime
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            shutil.copy(self.labels_filepath, os.path.join(backup_dir, f'{backup_prefix}{current_datetime}'))
        
            # Remove the oldest backup if more than three backups exist
            while len(backups) >= 3:
                oldest_backup = backups.popleft()
                os.remove(os.path.join(backup_dir, oldest_backup))
                        
            # Atomically move the temporary file to the target file
            shutil.move(labels_filepath_temp, self.labels_filepath)
    
        else:
            # Write the dictionary directly to the target file
            with open(self.labels_filepath, 'w') as temp_file:
                json.dump(self.labels, temp_file, indent=4, cls=ul._NoIndentEncoder)
        print("- saving labels")

        
        ## save yaml
        if not self.tag in self.attributes["project_data"]["progress"]:
            self.attributes["project_data"]["progress"][self.tag ] = {}
        self.attributes["project_data"]["progress"][self.tag] = {
            "n_processed": len(self.labels),
            "current_idx": self.current.idx,
            "current_image": self.current.image_name,
            "current_image_folder": self.current.image_folder,
            }
        
        ul._save_yaml(self.attributes, os.path.join(self.root_dir, "attributes.yaml"))
        print("- saving progress")

class Project:
    """
    Initialize a phenopype project with a root directory path. Phenopype 
    will create the project folder at the provided location. 

    Parameters
    ----------
    rootdir: str
        path to root directory of the project where folder gets created
    overwrite: bool, optional
        overwrite option, if a given root directory already exist 
        (WARNING: also removes all folders inside)

    Returns
    -------
    project: project
        phenopype project
    """

    def __init__(
            self, 
            root_dir, 
            load=True, 
            overwrite=False,
            ask=True,
            ):

        ## set flags
        flags = make_dataclass(
            cls_name="flags",
            fields=[("load", bool, load), 
                    ("overwrite", bool, overwrite),
                    ("checked", bool, False),
                    ("ask", bool, ask),

                    ],
        )

        ## path conversion
        root_dir = root_dir.replace(os.sep, "/")
        root_dir = os.path.abspath(root_dir)

        print("--------------------------------------------")
        while True:
            if os.path.isdir(root_dir):
                if all(
                    [
                        "attributes.yaml" in os.listdir(root_dir),
                        "data" in os.listdir(root_dir),
                    ]
                ):
                    if flags.load and not flags.overwrite:
                        print(
                            "Found existing project root directory - loading from:\n"
                            + root_dir
                        )
                        break
                    elif not flags.load and flags.overwrite:
                        pass
                    elif flags.load and flags.overwrite:
                        print(
                            "Found existing phenopype project directory at:\n{}\n".format(
                                root_dir
                            )
                        )
                        # time.sleep(0.1)
                        query1 = input("overwrite (y/n)?")
                        if query1 in settings.confirm_options:
                            pass
                        else:
                            print(
                                'Aborted - project "{}" not overwritten'.format(
                                    os.path.basename(root_dir)
                                )
                            )
                            return
                    shutil.rmtree(root_dir, onerror=ul._del_rw)
                    os.makedirs(root_dir)
                    os.makedirs(os.path.join(root_dir, "data"))
                    print('\n"' + root_dir + '" created (overwritten)')
                    break
                else:
                    print(
                        "Directory is neither empty nor a valid phenopype directory - aborting."
                    )
                    return
            else:
                print(
                    "Creating a new phenopype project directory at:\n" + root_dir + "\n"
                )
                if flags.ask:
                    query2 = input("Proceed? (y/n)\n")
                else:
                    query2 = "y"
                if query2 in settings.confirm_options:
                    os.makedirs(root_dir)
                    os.makedirs(os.path.join(root_dir, "data"))
                    break
                else:
                    print('\n"' + root_dir + '" not created!')
                    return

        ## read directories
        dir_names_counted, dir_paths = os.listdir(os.path.join(root_dir, "data")), []
        for file_path in os.listdir(os.path.join(root_dir, "data")):
            dir_paths.append(os.path.join(root_dir, "data", file_path))

        ## global project attributes
        project_attributes_path = os.path.join(root_dir, "attributes.yaml")
        if not os.path.isfile(project_attributes_path):
            project_attributes = {
                "project_info": {
                    "date_created": datetime.today().strftime(settings.strftime_format),
                    "date_changed": datetime.today().strftime(settings.strftime_format),
                    "phenopype_version": __version__,
                },
                "project_data": {
                    "filenames": [],
                    "dirnames": [],
                    
                    },
            }
            ul._save_yaml(
                project_attributes, os.path.join(root_dir, "attributes.yaml")
            )
            print(
                '\nProject "{}" successfully created.'.format(
                    os.path.basename(root_dir)
                )
            )
        else:
            project_attributes = ul._load_yaml(project_attributes_path)
            
            if "filenames" in project_attributes["project_data"]:
                
                file_names_attr = project_attributes["project_data"]["filenames"]
                dir_names_attr = project_attributes["project_data"]["dirnames"]

                if len(dir_names_attr) == len(file_names_attr) == len(dir_names_counted):
                    flags.checked = True
                    print("\nChecks for directory completeness passed!")

                else:
                    print("\nWARNING: Number of images in existing project and in project attributes are not matching")

            else:
                print("\nWARNING: Potentially broken project attributes file - could not check for completeness.")
                
            if len(dir_names_counted) > 0:
                print('\nProject "{}" successfully loaded with {} images'.format(
                        os.path.basename(root_dir), len(dir_paths)
                    ))
            else:
                print('\nProject "{}" successfully loaded, but it didn\'t contain any images!'.format(
                        os.path.basename(root_dir)
                    ))

        print("--------------------------------------------")

        ## attach to instance
        self.root_dir = root_dir
        self.dir_paths = dir_paths
        if flags.checked:
            self.file_names = file_names_attr
            self.dir_names = dir_names_attr
        else:
            self.file_names = []
            self.dir_names = []
            
        ## add attributes
        self.attributes = project_attributes
        self.attributes_path = project_attributes_path
     

    def add_files(
        self,
        image_dir,
        filetypes=settings.default_filetypes,
        include=[],
        include_all=True,
        exclude=[],
        mode="copy",
        n_max=None,
        nested=False,
        image_format=None,
        recursive=False,
        overwrite=False,
        resize_factor=1,
        resize_max_dim=None,
        unique="path",
        **kwargs
    ):
        """
        Add files to your project from a directory, can look recursively. 
        Specify in- or exclude arguments, filetypes, duplicate-action and copy 
        or link raw files to save memory on the harddrive. For each found image,
        a folder will be created in the "data" folder within the projects root
        directory. If found images are in subfolders and "recursive==True", 
        the respective phenopype directories will be created with 
        flattened path as prefix. 
        
        E.g., with "raw_files" as folder with the original image files 
        and "phenopype_proj" as rootfolder:
        
        - raw_files/file.jpg ==> phenopype_proj/data/file.jpg
        - raw_files/subdir1/file.jpg ==> phenopype_proj/data/1__subdir1__file.jpg
        - raw_files/subdir1/subdir2/file.jpg ==> phenopype_proj/data/2__subdir1__subdir2__file.jpg
    
        Parameters
        ----------
        image_dir: str 
            path to directory with images
        filetypes: list or str, optional
            single or multiple string patterns to target files with certain endings.
            "settings.default_filetypes" are configured in settings.py: 
            ['jpg', 'JPG', 'jpeg', 'JPEG', 'tif', 'png', 'bmp']
        include: list or str, optional
            single or multiple string patterns to target certain files to include
        include_all (optional): bool,
            either all (True) or any (False) of the provided keywords have to match
        exclude: list or str, optional
            single or multiple string patterns to target certain files to exclude - 
            can overrule "include"
        recursive: (optional): bool,
            "False" searches only current directory for valid files; "True" walks 
            through all subdirectories
        unique: {"file_path", "filename"}, str, optional:
            how to deal with image duplicates - "file_path" is useful if identically 
            named files exist in different subfolders (folder structure will be 
            collapsed and goes into the filename), whereas filename will ignore 
            all similar named files after their first occurrence.
        mode: {"copy", "mod", "link"} str, optional
            how should the raw files be passed on to the phenopype directory tree: 
            "copy" will make a copy of the original file, "mod" will store a 
            .tif version of the orginal image that can be resized, and "link" 
            will only store the link to the original file location to attributes, 
            but not copy the actual file (useful for big files, but the orginal 
            location needs always to be available)
        overwrite: {"file", "dir", False} str/bool (optional)
            "file" will overwrite the image file and modify the attributes accordingly, 
            "dir" will  overwrite the entire image directory (including all meta-data
            and results!), False will not overwrite anything
        ext: {".tif", ".bmp", ".jpg", ".png"}, str, optional
            file extension for "mod" mode
        resize_factor: float, optional
            
        kwargs: 
            developer options
        """

        # kwargs
        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("mode", str, mode),
                ("recursive", bool, recursive),
                ("overwrite", bool, overwrite),
                ("resize", bool, False),
            ],
        )

        if resize_factor < 1:
            flags.resize = True
            if not flags.mode == "mod":
                flags.mode = "mod"
                print('Resize factor <1 or >1 - switched to "mod" mode')

        ## path conversion
        image_dir = image_dir.replace(os.sep, "/")
        image_dir = os.path.abspath(image_dir)

        ## collect filepaths
        filepaths, duplicates = ul._file_walker(
            directory=image_dir,
            recursive=recursive,
            unique=unique,
            filetypes=filetypes,
            exclude=exclude,
            include=include,
            include_all=include_all,
        )

        ## feedback
        print("--------------------------------------------")
        print("phenopype will search for image files at\n")
        print(image_dir)
        print("\nusing the following settings:\n")
        print(
            "filetypes: "
            + str(filetypes)
            + ", include: "
            + str(include)
            + ", exclude: "
            + str(exclude)
            + ", mode: "
            + str(flags.mode)
            + ", recursive: "
            + str(flags.recursive)
            + ", resize: "
            + str(flags.resize)
            + ", unique: "
            + str(unique)
            + "\n"
        )
        
        ## subnsetting
        n_total_found = len(filepaths)
        if not n_max.__class__.__name__ == "NoneType":
            n_cut = min(n_max, n_total_found)
            filepaths = filepaths[:n_cut]
            n_max = str(n_max)
        else:
            n_max = "all"
            
        ## loop through files
        filenames = []
        for file_path in filepaths:
            
            ## image name and extension
            image_name = os.path.basename(file_path)
            image_name_stem = os.path.splitext(image_name)[0]
            image_ext = os.path.splitext(image_name)[1]
            filenames.append(image_name)

            ## generate folder paths by flattening nested directories; one
            ## folder per file
            relpath = os.path.relpath(file_path, image_dir)
            depth = relpath.count("\\")
            relpath_flat = os.path.dirname(relpath).replace("\\", "__")
            if depth > 0:
                subfolder_prefix = str(depth) + "__" + relpath_flat + "__"
            else:
                subfolder_prefix = str(depth) + "__"
                
            ## check if image exists
            if image_name in self.file_names:
                image_idx = self.file_names.index(image_name)
                dir_name = (subfolder_prefix + image_name_stem)
                dir_path = self.dir_paths[image_idx]
            else:
                dir_name = (subfolder_prefix + image_name_stem)
                dir_path = os.path.join(self.root_dir, "data", dir_name)
                
            ## make image-specific directories
            if os.path.isdir(dir_path):
                if flags.overwrite == False:
                    print(
                        "Found image "
                        + relpath
                        + " - "
                        + dir_name
                        + " already exists (overwrite=False)"
                    )
                    continue
                elif flags.overwrite in ["file", "files", "image", True]:
                    pass
                elif flags.overwrite == "dir":
                    shutil.rmtree(
                        dir_path, ignore_errors=True, onerror=ul._del_rw
                    )
                    print(
                        "Found image "
                        + relpath
                        + " - "
                        + "phenopype-project folder "
                        + dir_name
                        + ' created (overwrite="dir")'
                    )
                    os.mkdir(dir_path)
            else:
                print(
                    "Found image "
                    + relpath
                    + " - "
                    + "phenopype-project folder "
                    + dir_name
                    + " created"
                )
                os.mkdir(dir_path)
                
            ## generate image attributes
            image_data_original = ul._load_image_data(file_path)
            image_data_phenopype = {
                "date_added": datetime.today().strftime(settings.strftime_format),
                "mode": flags.mode,
            }

            ## copy or link raw files
            if flags.mode == "copy":
                image_phenopype_path = os.path.join(
                    self.root_dir, "data", dir_name, image_name_stem + "_copy" + image_ext,
                )
                shutil.copyfile(file_path, image_phenopype_path)
                image_data_phenopype.update(
                    ul._load_image_data(
                        image_phenopype_path, path_and_type=False
                    )
                )

            elif flags.mode == "mod":
                image = utils.load_image(file_path)
                image = ul._resize_image(
                    image, 
                    factor=resize_factor, 
                    max_dim=resize_max_dim
                    )
                if not image_format.__class__.__name__ == "NoneType":
                    if not "." in image_format:
                        ext = "." + image_format
                else:
                    ext = image_ext
                image_phenopype_path = os.path.join(
                    self.root_dir, "data", dir_name, image_name_stem + "_mod" + ext,
                )
                if all([
                        os.path.isfile(image_phenopype_path),
                        flags.overwrite in ["file", "files", "image", True]
                        ]):
                    print(
                        "Found image "
                        + relpath
                        + " - "
                        + "overwriting image and attributes in "
                        + dir_name
                        + ' (overwrite={})'.format(flags.overwrite)
                    )
                cv2.imwrite(image_phenopype_path, image)
                image_data_phenopype.update(
                    {"resize": flags.resize, "resize_factor": resize_factor,}
                )
                image_data_phenopype.update(
                    ul._load_image_data(
                        image_phenopype_path, path_and_type=False
                    )
                )

            elif flags.mode == "link":                
                image_phenopype_path = os.path.relpath(file_path, start=dir_path)
                image_data_phenopype.update(
                    ul._load_image_data(
                        image_path=file_path, 
                        image_rel_path=image_phenopype_path,
                        path_and_type=True
                        )
                    )
                if all([
                        os.path.isfile(os.path.join(dir_path, "attributes.yaml")),
                        flags.overwrite in ["file", "files", "image", True]
                        ]):
                    print(
                        "Found image "
                        + relpath
                        + " - "
                        + "overwriting attributes in "
                        + dir_name
                        + ' (overwrite={})'.format(flags.overwrite)
                    )

            ## write attributes file
            attributes = {
                "image_original": image_data_original,
                "image_phenopype": image_data_phenopype,
            }
            ul._save_yaml(
                attributes, os.path.join(dir_path, "attributes.yaml")
            )

        ## list dirs in data and add to project-attributes file in project root
        project_attributes = ul._load_yaml(
            os.path.join(self.root_dir, "attributes.yaml")
        )
        if any([flags.overwrite,
                project_attributes["project_data"].__class__.__name__ in ["CommentedSeq","list"]]):
            project_attributes["project_data"] = {}
            
        project_attributes["project_data"]["filenames"] = filenames
        project_attributes["project_data"]["dirnames"] =  os.listdir(os.path.join(self.root_dir, "data"))
        
        ul._save_yaml(
            project_attributes, 
            os.path.join(self.root_dir, "attributes.yaml")
        )

        ## add dirlists to project object (always overwrite)
        dir_names = os.listdir(os.path.join(self.root_dir, "data"))
        dir_paths = []
        for dir_name in dir_names:
            dir_paths.append(os.path.join(self.root_dir, "data", dir_name))
        self.dir_names = dir_names
        self.dir_paths = dir_paths

        print("\nFound {} files - using {}".format(n_total_found, n_max))
        print("--------------------------------------------")

    def add_config(
        self,
        tag,
        template_path,
        subset=[],
        interactive=False,
        image_number=1,
        overwrite=False,
        **kwargs
    ):
        """
        Add pype configuration presets to all image folders in the project, either by using
        the templates included in the presets folder, or by adding your own templates
        by providing a path to a yaml file. Can be tested and modified using the 
        interactive flag before distributing the config files.

        Parameters
        ----------

        tag: str
            tag of config-file. this gets appended to all files and serves as and
            identifier of a specific analysis pipeline
        template_path: str, optional
            path to a template or config-file in yaml-format
        subset: list, optional
            provide a list of images or folders in the project where the config should be 
            added. useful if you do not want to modify the config for all images
        interactive: bool, optional
            start a pype and modify template before saving it to phenopype directories
        interactive_image: str, optional
            to modify pype config in interactive mode, select image from list of images
            (directory names) already included in the project. special flag "first" is 
            default and takes first image in "data" folder. 
        overwrite: bool, optional
            overwrite option, if a given pype config-file already exist
        kwargs: 
            developer options
        """

        # =============================================================================
        ## setup

        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("interactive", bool, interactive), 
                ("overwrite", bool, overwrite),
                ],
        )

        ## apply subset if given 
        if len(subset) > 0:
            indices = [i for i, item in enumerate(self.dir_paths) if item in set(subset)]
            if len(indices) == 0:
                indices = [i for i, item in enumerate(self.file_names) if item in set(subset)]
            if len(indices) == 0:
                print("No directories or files found using given subset - aborting.")
                return
            flags.dir_paths = [self.dir_paths[i] for i in indices]
        else: 
            flags.dir_paths = self.dir_paths

        ## check tag sanity
        ul._check_pype_tag(tag)
        
        # =============================================================================
        ## interactive template modification
        
        if flags.interactive:
            if len(flags.dir_paths) > 0:
                dir_path = flags.dir_paths[image_number - 1]
            else:
                print(
                    "Project contains no images - could not add config files in interactive mode."
                )
                return

            if os.path.isdir(dir_path):
                container = ul._load_project_image_directory(dir_path)
                container.dir_path = os.path.join(self.root_dir, "_template-mod")
            else:
                print("Could not enter interactive mode - invalid directory.")
                return

            if not os.path.isdir(container.dir_path):
                os.mkdir(container.dir_path)

            config_path = utils.load_template(
                template_path=template_path,
                dir_path=container.dir_path,
                tag="template-mod",
                overwrite=True,
                ret_path=True,
            )

            ## run pype
            p = Pype(container, tag="template-mod", config_path=config_path,)
            template_path = p.config_path

        # =============================================================================
        ## save config to each directory
        
        if os.path.isfile(template_path):
            for dir_path in flags.dir_paths:
                utils.load_template(
                    template_path=template_path,
                    dir_path=dir_path,
                    tag=tag,
                    overwrite=flags.overwrite,
                )
            _config.template_path_current = None
            _config.template_loaded_current = None
            
        else:
            print("Could not find template_path")
            return

        if flags.interactive:
            q = input("Save modified template? ")
            if q in settings.confirm_options:
                template_dir = os.path.join(self.root_dir, "templates")
                if not os.path.isdir(template_dir):
                    os.mkdir(template_dir)
                q = input("Enter template file name: ")
                if q.endswith(".yaml"):
                    ext = ""
                else:
                    ext = ".yaml"
                template_file_name = q + ext

                template_save_path = os.path.join(template_dir, template_file_name)
                if ul._save_prompt("template", template_save_path, False):
                    ul._save_yaml(p.config, template_save_path)
                      
    
    def add_model(
        self,
        model_id,
        model_path,
        model_config_path=None,
        model_type="segmentation",
        overwrite=False,
        copy=True,
        **kwargs
    ):
        """
        Add a deep learning model. 

        Parameters
        ----------

        overwrite: bool, optional
            overwrite option, if a given pype config-file already exist
        template: bool, optional
            should a template for reference detection be created. with an existing 
            template, phenopype can try to find a reference card in a given image,
            measure its dimensions, and adjust pixel-to-mm-ratio and colour space
        """
        # =============================================================================
        # setup

        ## set flags
        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("overwrite", bool, overwrite), 
                ],
        )

        ## check model path
        if model_path.__class__.__name__ == "str":
            if os.path.isfile(model_path):
                pass
            else:
                print("Did not find model {}.".format(model_path))
                return
        else:
            print("Wrong input - need path to a model.")
            return

        
        while True:
            
            ## create reference attributes
            model_info = {
                "model_path": model_path,
                "model_name": os.path.splitext(os.path.basename(model_path))[0],
                "model_type": model_type,
            }
                 
            if model_config_path:
                if os.path.isfile(model_config_path):
                    model_info["model_config_path"] = model_config_path
                
            model_info["date_added"] = datetime.today().strftime(settings.strftime_format)

            ## load project attributes and temporarily drop project data list to
            ## be reattched later, so it is always at then end of the file
            model_dict = {}
            project_attributes = ul._load_yaml(
                os.path.join(self.root_dir, "attributes.yaml")
            )
            if "project_data" in project_attributes:
                project_data = project_attributes["project_data"]
                project_attributes.pop("project_data", None)
            if "models" in project_attributes:
                model_dict = project_attributes["models"]
                
            model_dict[model_id] = model_info
            project_attributes["models"] = model_dict
            project_attributes["project_data"] = project_data

            ul._save_yaml(
                project_attributes, os.path.join(self.root_dir, "attributes.yaml")
            )

            ul._print("\nSaved model info to project attributes.")
            break

        if flags.activate == True:
            _config.active_model_path = model_path
            ul._print('- adding model to {}'.format(model_id))
        else:
            ul._print(
                "- could not add model (overwrite=False/activate=False)"
            )

    def add_reference_template(self, image_path, reference_id, template=True, overwrite=False, **kwargs):
        """
        Add pype configuration presets to all project directories. 
    
        Parameters
        ----------
        image_path: str
            Path to the template image, either a file link, project directory, 
            or int (idx in project directories).
        reference_id: str
            Unique identifier for the reference.
        activate: bool, optional
            Whether to activate this reference across the project.
        template: bool, optional
            Whether to create a template for reference detection.
        overwrite: bool, optional
            Whether to overwrite existing files if they exist.             
            
        """
        reference_folder_path = os.path.join(self.root_dir, "reference")
        os.makedirs(reference_folder_path, exist_ok=True)
    
        # Manage reference and template files
        template_path = os.path.join(reference_folder_path, f"{reference_id}_template.tif")
        
        if type(image_path) == int:
            image_path = self.dir_paths[image_path]
        
        reference_image = utils.load_image(image_path)
    
        if os.path.isfile(template_path) and not overwrite:
            print("File already exists, not saving (overwrite=False)")
        else:
            
            ## get px ratio and mask
            annotations = kwargs.get("annotations", preprocessing.create_reference(reference_image))
            annotations = preprocessing.create_mask(reference_image, annotations=annotations)
            coords = annotations['mask']['a']['data']['mask'][0]
            
            ## save template image
            template = reference_image[coords[0][1]:coords[2][1], coords[0][0]:coords[1][0]]
            cv2.imwrite(template_path, template)
            print(f"Saved file: {template_path} (overwrite={overwrite})")
    
            # Save reference information to project attributes
            reference_template_info = {
                "source_image_path": os.path.abspath(image_path),
                "template_path": os.path.abspath(template_path),
                "template_px_ratio": annotations[settings._reference_type]["a"]["data"][settings._reference_type][0],
                "unit": annotations[settings._reference_type]["a"]["data"][settings._reference_type][1],
                "date_added": datetime.today().strftime(settings.strftime_format),
            }
            
            ## load project attributes and temporarily drop project data list to
            ## be reattched later, so it is always at then end of the file
            project_attributes = ul._load_yaml(os.path.join(self.root_dir, "attributes.yaml"))
            if "project_data" in project_attributes:
                project_data = project_attributes["project_data"]
                project_attributes.pop("project_data", None)
            project_attributes.setdefault("reference_templates", {})[reference_id] = reference_template_info
            project_attributes["project_data"] = project_data
            ul._save_yaml(project_attributes, os.path.join(self.root_dir, "attributes.yaml"))
            print(f"Updated project attributes with reference {reference_id}")       
            
    @ul.deprecation_warning(new_func=add_reference_template)
    def add_reference(self, template):
        pass
                
    def check_files(
            self, 
            feedback=True,
            image_links=False,
            new_dir=None,
            ret_missing=False,
            ):
        """
        Check all project files for completeness by comparing the images in the
        "data" folder to the file list the project attributes. Will attempt to 
        fix discrepancies, but ask for feedback first.

        Parameters
        ----------
        feedback : bool, optional
            Asks whether project attributes should be updated. The default is True.
        image_links : bool, optional
            checks whether image can be loaded from path, otherwise tries to load 
            from original filepath (will ask first). The default is False.
            

        Returns
        -------
        None.

        """
        
        # =============================================================================
        # setup

        ## set flags
        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("image_links", bool, image_links), 
                ("feedback", bool, feedback),
                ("check", bool, None),
                ],
        )

                
        ## load filenames
        if "filenames" in self.attributes["project_data"]:
            filenames = self.attributes["project_data"]["filenames"]
            dirnames = self.attributes["project_data"]["dirnames"]
        elif self.attributes["project_data"].__class__.__name__ in ["list","CommentedSeq"]:
             dirnames = self.attributes["project_data"]
             filenames = []
             self.attributes["project_data"] = {}
        else:
            print("could not read project attributes file!")
            return
        
        ## make lists
        filenames_check, dirnames_check, new_dir_files = [], [], None

        if os.path.isdir(str(new_dir)):
            new_dir_files = os.listdir(new_dir)
            
        for dirpath in self.dir_paths:
            attributes_path = os.path.join(dirpath, "attributes.yaml")
            attributes = ul._load_yaml(attributes_path)
            filename = attributes["image_original"]["filename"]
            filenames_check.append(filename)
            dirnames_check.append(os.path.basename(dirpath))
            
            if flags.image_links:
                filepath = attributes["image_phenopype"]["filepath"]
                if attributes["image_phenopype"]["mode"] == "link":
                    if not os.path.isfile(os.path.join(dirpath, filepath)):
                        ul._print("Could not find image(s) saved or linked to phenopype project:{}".format(filename))
                        if new_dir_files:
                            ul._print("- attempting to relink from {}".format(os.path.basename(new_dir)))
                            if filename in new_dir_files:
                                new_file_path = os.path.join(new_dir, filename)
                                attributes["image_phenopype"]["filepath"] = os.path.relpath(new_file_path, dirpath) 
                                ul._save_yaml(attributes, attributes_path)
                                ul._print("File found and successfully relinked!", lvl=1)
                            else:
                                ul._print("File not found in provided new_dir!")
                        else:
                            ul._print("No directory provided for relinking!", lvl=1)
                            if flags.check.__class__.__name__ == "NoneType":
                                flags.check = input("\nCheck original filepath and relink if possible [also for all other broken paths] (y/n)?\n")
                            if flags.check in settings.confirm_options:
                                filepath = attributes["image_original"]["filepath"]
                                if os.path.isfile(filepath):
                                    attributes["image_phenopype"]["filepath"] = os.path.relpath(filepath, dirpath)
                                    ul._print("Re-linking successful - saving new path to attributes!", lvl=1)
                                    ul._save_yaml(attributes, attributes_path)
    
                            else:
                                ul._print("File not found - could not re-link!")

        filenames_unmatched, dirnames_unmatched = [], []
        
        for filename in filenames:
            if not filename in filenames_check:
                filenames_unmatched.append(filename)

        for dirname in dirnames:
            if not filename in dirnames_check:
                dirnames_unmatched.append(filename)  
                
        if len(filenames_unmatched) > 0:
            ul._print(filenames_unmatched)
            ul._print("\n")
            ul._print("--------------------------------------------")
            ul._print("phenopype found {} files in the data folder, but {} from the project attributes are unaccounted for.".format(
                len(filenames_check),
                len(filenames_unmatched)))
            time.sleep(1)
            if flags.feedback:
                check = input("update project attributes (y/n)?")
            else:
                check = "y"
        elif len(filenames) == 0:
            ul._print("\n")
            ul._print("--------------------------------------------")
            ul._print("phenopype found {} files in the data folder, but 0 are listed in the project attributes.".format(len(filenames_check)))
            time.sleep(1)
            if flags.feedback:
                check = input("update project attributes (y/n)?")
            else:
                check = "y"
        elif len(filenames) < len(filenames_check):
            ul._print("\n")
            ul._print("--------------------------------------------")
            ul._print("phenopype found {} files in the data folder, but only {} are listed in the project attributes.".format(
                len(filenames_check),
                len(filenames)
                ))
            time.sleep(1)
            if flags.feedback:
                check = input("update project attributes (y/n)?")
            else:
                check = "y"
        else:
            ul._print("All checks passed - numbers in data folder and attributes file match.", lvl=2)
            return

        if check in settings.confirm_options:
            
            self.attributes["project_data"].pop("filenames", None)
            self.attributes["project_data"].pop("dirnames", None)
            self.attributes["project_data"]["filenames"] = filenames_check
            self.attributes["project_data"]["dirnames"] = dirnames_check
            self.file_names = filenames_check
            self.dir_names = dirnames_check
            
            ul._save_yaml(self.attributes, self.attributes_path)
            
            ul._print("project attributes updated; now has {} files".format(len(filenames_check)), lvl=2)
        else:
            ul._print("project attributes not updated", lvl=2)
        print("--------------------------------------------")    
        
        
        if ret_missing:
            return filenames_unmatched
        

    def collect_results(
            self,
            tag, 
            files, 
            folder="", 
            aggregate_csv=True,
            overwrite=False,
            **kwargs
            ):

        """
        Collect canvas from each folder in the project tree. Search by 
        name/safe_suffix (e.g. "v1").

        Parameters
        ----------
        tag : str
            pype tag / save_suffix
        files : str 
            list results (canvas, annotation-file, or csv-files) to be aggregated to a single directory 
            at the project root
        folder : str, optional {default: <annotation-name>_<tag> }
            folder in the root directory where the aggregated results are stored
        aggregate_csv : bool, optional 
            concatenate csv files of the same type and store in results folder instead
            of each file separately
        overwrite : bool, optional
            should the results be overwritten
    
        """
        
        # =============================================================================
        # setup
                
        ## set flags
        flags = make_dataclass(
            cls_name="flags", 
            fields=[
                ("folder", str, folder),
                ("aggregate_csv", bool, aggregate_csv),
                ("overwrite", bool, overwrite),
                ])

        ## create results folder
        results_dir = os.path.join(self.root_dir, "results")
        os.makedirs(results_dir, exist_ok=True)   
            
        # =============================================================================
        # execute

        ## search string
        if not files.__class__.__name__ == "list":
            files = [files]

        ## exclude strings
        exclude = kwargs.get("exclude", [])
        if not exclude.__class__.__name__ == "NoneType":
            if exclude.__class__.__name__ == "str":
                exclude = [exclude]
        exclude=exclude + ["pype_config", "attributes"]

        ## search
        for file in files:
            search_string = [file, tag]
            results, duplicates = ul._file_walker(
                os.path.join(self.root_dir, "data"),
                recursive=True,
                include=search_string,
                include_all=True,
                exclude=exclude,
            )
            
            if len(results) == 0:
                results = ["no-results"]
                
            ## search feedback
            print("file \"{}\": found {} results in {} project folders".format(
                file, len(results), len(self.dir_names)))

            ## save to csv
            if all([flags.aggregate_csv,
                    results[0] != "no-results",
                    results[0].endswith(".csv")]):
                
                result_list = []
                for path in results:
                    result_list.append(pd.read_csv(path))
                result = pd.concat(result_list)
                csv_name = file + "_" + tag + ".csv"
                csv_path = os.path.join(results_dir, csv_name)
                
                ## overwrite check
                if os.path.isfile(csv_path) and not flags.overwrite:
                    print("Results file {} not saved: already exists (overwrite=False)".format(csv_name))
                else:
                    print("Saving file {}".format(csv_name))
                    result.to_csv(csv_path, index=False)
                
            ## copy files to subfolders
            elif results[0] != "no-results":
                if len(folder)==0:
                    folder = file + "_" + tag
                folder_path = os.path.join(results_dir, folder)
                os.makedirs(folder_path, exist_ok=True)
                for old_path in results:
                    new_file = (
                        os.path.basename(os.path.dirname(old_path))
                        + "_"
                        + os.path.basename(old_path)
                    )
                    new_path = os.path.join(folder_path, new_file)

                    ## overwrite check
                    if os.path.isfile(new_path) and flags.overwrite == False:
                        print(new_file + " not saved - file already exists (overwrite=False).")
                    else:
                        shutil.copyfile(old_path, new_path)
                    
            
    def copy_tag(
            self, 
            tag_src, 
            tag_dst, 
            copy_annotations=True, 
            copy_config=True,
            overwrite=False, 
            **kwargs):
        """
        Make a copy of data generated under a specific tag and save it under a 
        new tag - e.g.: 
            
            annotations_v1.json ==> annotations_v2.json
            pype_config_v1.yaml ==> pype_config_v2.yaml

        Parameters
        ----------
        tag_src : str
            name of tag to be copied (source tag)
        tag_dst : str
            name of new tag (destination tag)
        copy_annotations : bool, optional
            copy annotations file. The default is True.
        copy_config : bool, optional
            copy config file. The default is True.
        overwrite : bool, optional
             overwrites if tag exists. The default is False.
        kwargs: 
            developer options

        Returns
        -------
        None.

        """
   
        ## go through project directories
        for directory in self.dir_paths:
            dir_name = os.path.basename(directory)

            if copy_annotations:
                annotations_path = os.path.join(
                    self.root_dir, "data", dir_name, "annotations_" + tag_src + ".json"
                )
                new_annotations_path = os.path.join(
                    self.root_dir, "data", dir_name, "annotations_" + tag_dst + ".json"
                )
                
                if os.path.isfile(annotations_path): 
                    if ul._overwrite_check_file(new_annotations_path, overwrite):
                        shutil.copyfile(annotations_path, new_annotations_path)
                        print("Copy annotations for {}".format(dir_name))
                else:
                    print("Missing annotations for {} - skipping".format(dir_name))

                    
            if copy_config:
                config_path = os.path.join(
                    self.root_dir, "data", dir_name, "pype_config_" + tag_src + ".yaml"
                )
                new_config_path = os.path.join(
                    self.root_dir, "data", dir_name, "pype_config_" + tag_dst + ".yaml"
                )
                
                if os.path.isfile(config_path): 
                    if ul._overwrite_check_file(new_config_path, overwrite):
                        shutil.copyfile(config_path, new_config_path)
                        print("Copy config for {}".format(dir_name))
                else:
                    print("Missing config for {} - skipping".format(dir_name))

                
    def edit_config(
            self, 
            tag, 
            target, 
            replacement, 
            subset=[],
            **kwargs):
        """
        Add or edit functions in all configuration files of a project. Finds and
        replaces single or multiline string-patterns. Ideally this is done via 
        python docstrings that represent the parts of the yaml file to be replaced.
                
        Parameters
        ----------

        tag: str
            tag (suffix) of config-file (e.g. "v1" in "pype_config_v1.yaml")
        target: str
            string pattern to be replaced. should be in triple-quotes to be exact
        replacement: str
            string pattern for replacement. should be in triple-quotes to be exact
        subset: list, optional
            provide a list of images or folders in the project where the config should be 
            modified. useful if you do not want to modify the config for all images
        """

        # =============================================================================
        ## setup
        
        ## kwargs and setup
        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("checked", bool, False), 
                ],
        )

        ## apply subset if given 
        if len(subset) > 0:
            indices = [i for i, item in enumerate(subset) if item in set(self.dir_paths)]
            if len(indices) == 0:
                indices = [i for i, item in enumerate(subset) if item in set(self.file_names)]
            if len(indices) == 0:
                print("No directories or files found using given subset - aborting.")
                return
            flags.dir_paths = [self.dir_paths[i] for i in indices]
        else: 
            flags.dir_paths = self.dir_paths
            
        # =============================================================================
        ## go through project directories
        
        for directory in flags.dir_paths:
            dir_name = os.path.basename(directory)

            ## get config path
            config_path = os.path.join(
                self.root_dir, "data", dir_name, "pype_config_" + tag + ".yaml"
            )

            ## open config-file
            if os.path.isfile(config_path):
                with open(config_path, "r") as config_text:
                    config_string = config_text.read()
            else:
                print("Did not find config file to edit - check provided tag/suffix.")
                return
            ## string replacement
            new_config_string = config_string.replace(target, replacement)

            ## show user replacement-result and ask for confirmation
            if flags.checked == False:
                print(new_config_string)
                check = input(
                    "This is what the new config may look like (can differ between files) - proceed?"
                )

            ## replace for all config files after positive user check
            if check in settings.confirm_options:
                flags.checked = True
                with open(config_path, "w") as config_text:
                    config_text.write(new_config_string)

                print("New config saved for " + dir_name)
            else:
                print("User check failed - aborting.")
                break
                    
            

    def export_zip(
            self, 
            tag=None, 
            results=False,
            models=False, 
            images=False, 
            exports=False,
            save_dir=None,
            overwrite=True,
            **kwargs
            ):
        """
    
        Parameters
        ----------
        tag: str, optional
            Export only files associated with a specified tag. The default is None.
        images : bool, optional
            Don't include images from the data folder. The default is True.

        Returns
        -------
        None.

        """


            
            
        # =============================================================================
        # setup

        ## set flags
        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("tag", str, tag), 
                ("results", bool, results),
                ("models", bool, models),
                ("images", bool, images),
                ("exports", bool, exports),
                ("overwrite", bool, overwrite),
                ],
        )

        if kwargs.get("save_suffix"):
            save_suffix = "_" + kwargs.get("save_suffix")
        else:
            save_suffix = ""
            
        ## construct save path
        root_dir = os.path.basename(self.root_dir) 
        if save_dir.__class__.__name__ == "NoneType":
            save_dir = self.root_dir
            
        save_path = os.path.join(save_dir, root_dir + save_suffix + ".zip")
        
        if os.path.isfile(save_path) and flags.overwrite:
            os.remove(save_path)

        # =============================================================================
        # execute

        ## start zip process
        with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zip:
    
            ## project root dir
            zip.write(os.path.join(self.root_dir), root_dir)    
    
            ## project attributes
            zip.write(
                os.path.join(self.root_dir, "attributes.yaml"), 
                os.path.join(root_dir, "attributes.yaml")
                )    
            
            ## data folder
            data_dir_abspath = os.path.join(self.root_dir, "data")       
            data_dir_relpath = os.path.join(root_dir, "data")       
            zip.write(data_dir_abspath, data_dir_relpath)    
            
            ## loop thropugh data folder               
            for folder in os.listdir(data_dir_abspath):
    
                folder_abspath = os.path.join(data_dir_abspath, folder)
                folder_relpath = os.path.join(data_dir_relpath, folder)
                zip.write(folder_abspath, folder_relpath)                            
    
                for file in os.listdir(folder_abspath):
                    
                    file_name, file_ext  = os.path.splitext(file)
                    
                    if not file in ["attributes.yaml"]:
                        if not flags.tag.__class__.__name__ == "NoneType":
                            test_pattern = file_name[-len(flags.tag):len(file_name)]
                            if not flags.tag == test_pattern:
                                continue        
                        if file_ext.strip(".") in settings.default_filetypes:
                            if not flags.images:
                                continue
                        if file_ext.strip(".") == "csv" and flags.exports == False:
                            continue
                        
                    file_abspath = os.path.join(data_dir_abspath, folder, file)
                    file_relpath = os.path.join(data_dir_relpath, folder, file)
                    zip.write(file_abspath, file_relpath)                            
                    
            if flags.models:
                
                models_dir_abspath = os.path.join(self.root_dir, "models")       
                models_dir_relpath = os.path.join(root_dir, "models") 
                zip.write(models_dir_abspath, models_dir_relpath)    
                
                for file in os.listdir(models_dir_abspath):
                    
                    file_abspath = os.path.join(models_dir_abspath, file)
                    file_relpath = os.path.join(models_dir_relpath, file)
                    zip.write(file_abspath, file_relpath)                      
    
            if flags.results:
           
                results_dir_abspath = os.path.join(self.root_dir, "results")       
                results_dir_relpath = os.path.join(root_dir, "results")       
                zip.write(results_dir_abspath, results_dir_relpath)    
                
                ## loop thropugh data folder               
                for folder in os.listdir(results_dir_abspath):
                        
                    folder_abspath = os.path.join(results_dir_abspath, folder)
                    folder_relpath = os.path.join(results_dir_relpath, folder)
                    zip.write(folder_abspath, folder_relpath)                            
        
                    if os.path.isdir(os.path.join(results_dir_abspath, folder)):
                        for file in os.listdir(folder_abspath):
                            
                            file_name, file_ext  = os.path.splitext(file)
                                                    
                            if file_ext.strip(".") in settings.default_filetypes:
                                if not flags.images:
                                    continue
                            
                            file_abspath = os.path.join(results_dir_abspath, folder, file)
                            file_relpath = os.path.join(results_dir_relpath, folder, file)
                            zip.write(file_abspath, file_relpath)   
    
    def export_training_data(
            self, 
            tag,
            method,
            params={},
            folder=None, 
            annotation_id=None, 
            overwrite=False, 
            img_size=224,
            parameters=None,
            **kwargs):
        """
        

        Parameters
        ----------
        tag : str
            The Pype tag from which training data should be extracted.
        method :  {"ml-morph"} str
            For which machine learning framwork should training data be created.
            Currently instructions for the following architectures are supported:
            
            - "ml-morph" - Machine-learning tools for landmark-based morphometrics 
              https://github.com/agporto/ml-morph
            - "keras-cnn-semantic" - Images and masks to be used for training an
              image segmentation model in Keras
            
        folder : str
            Name of the folder under "root/training_data" where the formatted 
            training data will be stored under.
        annotation_id : str, optional
            select a specific annotation id. The default is None.
        overwrite : bool, optional
            Should an existing training data set be overwritten. The default is False.

        Returns
        -------
        None.

        """
        
        # =============================================================================
        # setup
        
        flags = make_dataclass(
            cls_name="flags", fields=[
                ("overwrite", bool, overwrite),
                ]
        )
    
        if annotation_id.__class__.__name__ == "NoneType":
            print("No annotation id set - will use last one in annotations file.")
            time.sleep(1)
        if folder.__class__.__name__ == "NoneType":
            training_data_path = os.path.join(self.root_dir, "training_data", tag)
        else:
            assert os.path.exists(folder), "root folder for training images does not exist"
            training_data_path = folder
    
        if not os.path.isdir(training_data_path):
            os.makedirs(training_data_path)
            print("Created " + training_data_path)
            
        params_all = {
            "ml-morph": {
                "export_mask": False,
                "flip_y": False,
                "mask_id": None,
                "resize_factor": 1,
                },
            "generic-binary-mask" : {
                "local": True,
                "mode": "largest", 
                "resize": 1,
                "set_dim": None,
                "ext": "tif",
                },
            "keras-cnn-semantic": {
                }
            }

        params_all = copy.deepcopy(params_all)
        params_all[method].update(params)

        # =============================================================================
        ## ml-morph
        
        if method=="ml-morph":

            annotation_type = settings._landmark_type
            df_summary = pd.DataFrame()
            file_path_save = os.path.join(training_data_path, "landmarks_ml-morph_" + tag + ".csv")

            if not ul._overwrite_check_file(file_path_save, flags.overwrite):
                return
            
            if not parameters.__class__.__name__ == "NoneType":
                params["ml-morph"].update(parameters["ml-morph"])
                
                
            flags.export_mask = params["ml-morph"]["export_mask"]
            flags.resize_factor = params["ml-morph"]["resize_factor"]
            flags.flip_y = params["ml-morph"]["flip_y"]

            if flags.export_mask:  
                img_dir = os.path.join(training_data_path, "images")
                if not os.path.isdir(img_dir):
                    os.makedirs(img_dir)
            
            for idx1, dirpath in enumerate(self.dir_paths, 1):
                            
                ## load data
                attributes = ul._load_yaml(os.path.join(dirpath, "attributes.yaml"))           
                annotations = export.load_annotation(os.path.join(dirpath, "annotations_" + tag + ".json"))       
                filename = attributes["image_original"]["filename"]
                image_height = attributes["image_phenopype"]["height"]
                
                print("Preparing training data for: ({}/{}) ".format(idx1, len(self.dir_paths)) + filename)
    
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
                lm_tuple_list = list(zip(*data))
                
                ## load image if masking or resizing
                if flags.export_mask or flags.resize_factor != 1:
                    image = utils.load_image(dirpath)
                    image_height, image_width = image.shape[0:2]

                    ## select last mask if no id is given
                    if params["ml-morph"]["mask_id"].__class__.__name__ == "NoneType":
                        mask_id = max(list(annotations[settings._mask_type].keys()))
                        
                    ## get bounding rectangle and crop image to mask coords
                    if flags.export_mask and settings._mask_type in annotations:
                        coords = annotations[settings._mask_type][mask_id]["data"][settings._mask_type][0]
                        rx, ry, rw, rh = cv2.boundingRect(np.asarray(coords, dtype="int32"))
                        image = image[
                            max(ry,0) : min(ry + rh, image_height), max(rx,1) : min(rx + rw, image_width)]
                        image_height = image.shape[0]

                        ## subtract top left coord from bounding box from all landmarks
                        lm_tuple_list[0] = tuple(c - rx for c in lm_tuple_list[0])
                        lm_tuple_list[1] = tuple(c - ry for c in lm_tuple_list[1])
                        
                    ## resize image or cropped image
                    if flags.resize_factor != 1:
                        image = ul._resize_image(image, factor=flags.resize_factor, interpolation="cubic")
                        image_height = int(image_height * flags.resize_factor)
                        
                        ## multiply all landmarks with resize factor
                        lm_tuple_list[0] = tuple(int(c * flags.resize_factor) for c in lm_tuple_list[0])
                        lm_tuple_list[1] = tuple(int(c * flags.resize_factor) for c in lm_tuple_list[1])
                    
                    ## save resized image or cropped image
                    utils.save_image(image, dir_path=img_dir, file_name=filename, overwrite=overwrite)
                    
                ## add to dataframe
                coord_row, colnames = list(), list()
                colnames.append("id")
                for  idx2, (x_coord, y_coord) in enumerate(zip(lm_tuple_list[0], lm_tuple_list[1]),1):
                    coord_row.append(x_coord)
                    if flags.flip_y:
                        coord_row.append(int((y_coord - image_height) * -1))
                    else: 
                        coord_row.append(y_coord)
                    colnames.append("X" + str(idx2))
                    colnames.append("Y" + str(idx2))
                df = pd.DataFrame([filename] + coord_row).transpose()
                df_summary = pd.concat([df_summary, df])
                
            ## save dataframe
            df_summary.set_axis(colnames, axis=1, inplace=True)
            df_summary.to_csv(file_path_save, index=False)
            
            
        # =============================================================================
        ## generic-binary-masks
        
        if method=="generic-binary-mask":
            
            ## get params
            params = params_all[method]
            
            ## make dirs
            image_dir = os.path.join(folder, "images")
            mask_dir = os.path.join(folder, "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            
            if params["mode"] == "largest":
                which = "max"
            
            for idx, dirpath in enumerate(self.dir_paths, 1):
                
                print(str(idx) + " / " + str(len(self.dir_paths)))
                
                annotation_path = os.path.join(dirpath, "annotations_" + tag + ".json")
                
                if os.path.isfile(annotation_path):
                    annotations = export.load_annotation(annotation_path)   
                    image = utils.load_image(dirpath)
                    attributes = ul._load_yaml(os.path.join(dirpath, "attributes.yaml"))           
                    file_name = attributes["image_original"]["filename"] 
                    try:
                        roi, mask = export.save_ROI(
                            image=image,
                            annotations=annotations,
                            which=which,
                            dir_path=None,
                            file_name=None,
                            annotation_type="contour",
                            min_dim=512,
                            counter=False,
                            training_data=True,
                            )
                
                        roi_saved = utils.save_image(roi, file_name, suffix="roi", ext=params["ext"], dir_path=image_dir)
                        mask_saved = utils.save_image(mask, file_name, suffix="roi", ext=params["ext"], dir_path=mask_dir)
                    except:
                        print("missing annotation? - skipping")
                else:
                    print(f"no annotations with tag {tag} - skipping")

        # =============================================================================
        ## keras-cnn-semantic
        
        if method=="keras-cnn-semantic":
            
            annotation_type = kwargs.get("annotation_type", "mask")
            
            img_dir = os.path.join(training_data_path, "images", "all")
            mask_dir = os.path.join(training_data_path, "masks", "all")
            
            if not os.path.isdir(img_dir):
                os.makedirs(img_dir)
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)  

            for idx, dirpath in enumerate(self.dir_paths, 1):
                            
                attributes = ul._load_yaml(os.path.join(dirpath, "attributes.yaml"))           
                annotations = export.load_annotation(os.path.join(dirpath, "annotations_" + tag + ".json"))       
                filename = attributes["image_original"]["filename"]               
                
                print("Preparing training data for: ({}/{}) ".format(idx, len(self.dir_paths)) + filename)

                shape = (attributes["image_original"]["width"], attributes["image_original"]["height"], 3)
                mask = np.zeros(shape, dtype="uint8")
                if not annotation_type in annotations:
                    print("No annotation of type {} found in dataset - aborting".format(annotation_type))
                    return
                if annotation_id.__class__.__name__ == "NoneType":
                    annotation_id = max(list(annotations[annotation_type].keys()))
                    
                mask = visualization.draw_contour(image=mask, annotations=annotations, contour_id=annotation_id, line_width=0, line_colour=1, fill=1)
                mask_resized = cv2.resize(mask, (img_size, img_size))
                
                image = utils.load_image(dirpath)
                image_resized = cv2.resize(image, (img_size, img_size))
                
                utils.save_image(image_resized, dir_path=img_dir, file_name=filename, ext="png")
                utils.save_image(mask_resized, dir_path=mask_dir, file_name=filename, ext="png")

            

class Pype(object):
    """
    The pype is phenopype’s core method that allows running all functions 
    that are available in the program’s library in sequence. Users can execute 
    the pype method on a file_path, an array, or a phenopype directory, which 
    always will trigger three actions:

    1. open the contained yaml configuration with the default OS text editor
    2. parse the contained functions and execute them in the sequence (exceptions
       will be passed, but returned for diagnostics)
    3. open a Python-window showing the processed image.
    
    After one iteration of these steps, users can evaluate the results and decide
    to modify the opened configuration file (e.g. either change function parameters or 
    add new functions), and run the pype again, or to terminate the pype and 
    save all results. The processed image, any extracted phenotypic information, 
    as well as the modified config-file is stored inside the image directory, or
    a user-specified directory. By providing unique names, users can store different
    pype configurations and the associated results side by side. 
    
    Parameters
    ----------

    image: array or str 
        can be either a numpy array or a string that provides the path to 
        source image file or path to a valid phenopype directory
    name: str
        name of pype-config - will be appended to all results files
    config_template: str, optional
        chose from list of provided templates  
        (e.g. ex1, ex2, ...)
    config_path: str, optional
        custom path to a pype template (needs to adhere yaml syntax and 
        phenopype structure)
    delay: int, optional
        time in ms to add between reload attemps of yaml monitor. increase this 
        value if saved changes in config file are not parsed in the first attempt.
    dir_path: str, optional
        path to an existing directory where all output should be stored
    skip: bool, optional
        skip directories that already have "name" as a suffix in the filename
    interactive: bool, optional
        don't open text editor or window, just apply functions and terminate
    max_dim: int, optional
        maximum dimension that window can have 
    kwargs: 
        developer options
    """

    def __init__(
        self,
        image_path,
        tag,
        skip=False,
        interactive=True,
        feedback=True,
        autoload=True,
        autosave=True,
        autoshow=True,
        log_ow=False,
        config_path=None,
        fix_names=True,
        load_contours=False,
        zoom_memory=True,
        debug=False,
        **kwargs
    ):

        # =============================================================================
        # INIT

        ## kwargs
        global window_max_dim
        window_max_dim = kwargs.get("window_max_dim")
        delay = kwargs.get("delay", 100)
        
        ## flags
        self.flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("debug", bool, debug),
                ("autoload", bool, autoload),
                ("autosave", bool, autosave),
                ("autoshow", bool, autoshow),
                ("feedback", bool, feedback),
                ("interactive", bool, interactive),
                ("fix_names", bool, fix_names),
                ("skip", bool, skip),
                ("zoom_memory", bool, zoom_memory),
                ("dry_run", bool, kwargs.get("dry_run", False)),
                ("terminate", bool, False),
            ],
        )
        
        if self.flags.debug:
            
            self.verbose = _config.verbose
            _config.verbose = True
            
        if image_path.__class__.__name__ == "str":
            image_path = os.path.abspath(image_path)
            
        # =============================================================================
        # LOGGING
            
        ## start logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
                        
        if self.flags.feedback:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_formatter = logging.Formatter('%(asctime)s: %(message)s', "%H:%M:%S")
            stdout_handler.setFormatter(stdout_formatter)
            self.logger.addHandler(stdout_handler)

        ## log file
        log_file_path = os.path.join(image_path, f"pype_logs_{tag}.log")
        if os.path.isfile(log_file_path) and log_ow:
            os.remove(log_file_path)
            
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # =============================================================================
        # CHECKS 

        ## check name, load container and config
        ul._check_pype_tag(tag)
        self._load_container(image_path=image_path, tag=tag)
        self._load_pype_config(image_path=image_path, tag=tag, config_path=config_path)

        # check version, load container and config
        if self.flags.dry_run:
            self._load_pype_config(image_path, tag, config_path)
            self._iterate(config=self.config, annotations=copy.deepcopy(settings._annotation_types),
                      execute=False, autoshow=False, feedback=True)
            return

        ## check whether directory is skipped
        if self.flags.skip:
            if self._check_directory_skip(
                tag=tag, skip_pattern=skip, dir_path=self.container.dir_path
            ):
                return
            
        # =============================================================================
        # FEEDBACK 
        
        startup_msg_list = []
        startup_msg_list.append(ul._pprint_fill_hbar(self.container.image_name, symbol="=", ret=True))
        self._log("info", startup_msg_list, 0, passthrough=True)
        
        # =============================================================================

        ## load existing annotations through container
        if self.flags.autoload:
            self._log("info", "Pype: AUTOLOAD", 0)
            with io.StringIO() as buffer, redirect_stdout(buffer):
                self.container.load(contours=load_contours)
                stdout = buffer.getvalue()
                self._log("info", stdout, 1)

        ## check pype config for annotations
        self._iterate(
            config=self.config,
            annotations=self.container.annotations,
            execute=False,
            autoshow=False,
            interactive=False,
        )

        ## final check before starting pype
        self._check_final()
        
        ## turn off autshow and file monitor in non-interactive mode
        if self.flags.interactive:
            self._start_file_monitor(delay=delay)
        else:
            self.flags.autoshow = False
            
        ## clear old zoom memory
        _config.gui_zoom_config = None

        # =============================================================================
        # PYPE LOOP

        ## run pype
        while True:

            ## pype restart flag
            _config.pype_restart = False

            ## refresh config
            if self.flags.interactive:

                ## to stop infinite loop without opening new window
                if not self.YFM.content:
                    self._log("debug", "Pype: STILL UPDATING CONFIG (no config)", 0)
                    cv2.destroyWindow("phenopype")
                    time.sleep(1)
                    self.YFM._stop()
                    self._start_file_monitor(delay=delay)
                    continue

                self.config = copy.deepcopy(self.YFM.content)

                if not self.config:
                    self._log("debug", "Pype: STILL UPDATING CONFIG (no config)", 0)
                    continue

            ## run pype config in sequence
            self._iterate(
                config=self.config,
                annotations=self.container.annotations,
                interactive=self.flags.interactive,
                autoshow=self.flags.autoshow,
            )

            ## terminate & cleanup
            if self.flags.interactive:
                if self.flags.terminate:
                    
                    ## stope file monitoring
                    if hasattr(self, "YFM"):
                        self.YFM._stop()
                        
                    ## reset zoom settings
                    if not self.flags.zoom_memory:
                        _config.gui_zoom_config = None
                    
                    ## add timestamp
                    self.config["config_info"]["date_last_modified"] = datetime.today().strftime(settings.strftime_format)
                    ul._save_yaml(self.config, self.config_path)
                        
                    ## feedback
                    self._log("info", "Pype: TERMINATE", 0)
                    break
                
            else:
                break
            
        # =============================================================================
        ## autosave
        if self.flags.autosave and self.flags.terminate:
            self._log("info", "Pype: AUTOSAVE", 0)
            if "export" not in self.config_parsed_flattened:
                export_list = []
            else:
                export_list = self.config_parsed_flattened["export"]
            with io.StringIO() as buffer, redirect_stdout(buffer):
                self.container.save(export_list=export_list)
                stdout = buffer.getvalue()
                self._log("info", stdout, 1)
                
        # =============================================================================
        # FEEDBACK 
        
        startup_msg_list = []
        startup_msg_list.append(ul._pprint_fill_hbar("END", symbol="=", ret=True))
        self._log("info", startup_msg_list, 0)
        
        # =============================================================================

        ## end logging
        logging.shutdown()
        
        if self.flags.debug:
            _config.verbose = self.verbose


    def _load_container(self, image_path, tag):
        if image_path.__class__.__name__ == "str":
            if os.path.isfile(image_path):
                image = utils.load_image(image_path)
                dir_path = os.path.dirname(image_path)
                self.container = utils.Container(
                    image=image,
                    dir_path=dir_path,
                    tag=tag,
                    file_prefix=os.path.splitext(os.path.basename(image_path))[0],
                    file_suffix=tag,
                    image_name=os.path.basename(image_path),
                )
            elif os.path.isdir(image_path):
                self.container = ul._load_project_image_directory(
                    dir_path=image_path, tag=tag,
                )
            else:
                raise FileNotFoundError(
                    'Could not find image or image directory: "{}"'.format(
                        os.path.dirname(image_path)
                    )
                )
        elif image_path.__class__.__name__ == "Container":
            self.container = copy.deepcopy(image_path)
        else:
            raise TypeError("Invalid input for image path (str required)")

    def _load_pype_config(self, image_path, tag, config_path):

        if config_path.__class__.__name__ == "NoneType":
            if os.path.isfile(image_path):
                image_name_stem = os.path.splitext(os.path.basename(image_path))[0]
                prepend = image_name_stem + "_"
            elif os.path.isdir(image_path):
                prepend = ""

            ## generate config path from image file or directory (project)
            config_name = prepend + "pype_config_" + tag + ".yaml"
            config_path = os.path.join(self.container.dir_path, config_name)

        ## load config from config path
        elif config_path.__class__.__name__ == "str":
            if os.path.isfile(config_path):
                pass
            # else:
            #     raise FileNotFoundError(
            #         "Could not read config file from specified config_path: \"{}\"".format(config_path))

        if os.path.isfile(config_path):
            self.config = ul._load_yaml(config_path)
            self.config_path = config_path

            if "template_locked" in self.config:
                if self.config["template_locked"] == True:
                    raise AttributeError(
                        'Attempting to load config from locked template - create config file using "load_template" first.'
                    )
        else:
            raise FileNotFoundError(
                'Could not find config file "{}" in image directory: "{}"'.format(
                    config_name, os.path.dirname(image_path)
                )
            )

    def _start_file_monitor(self, delay):

        if platform.system() == "Darwin":  # macOS
            subprocess.call(("open", self.config_path))
        elif platform.system() == "Windows":  # Windows
            os.startfile(os.path.normpath(self.config_path))
        else:  # linux variants
            subprocess.call(("xdg-open", self.config_path))

        self.YFM = ul._YamlFileMonitor(self.config_path, delay)
        self._log("debug", "Pype: starting config file monitor", 0)

    def _check_directory_skip(self, tag, skip_pattern, dir_path):

        ## skip directories that already contain specified files
        if skip_pattern.__class__.__name__ == "str":
            skip_pattern = [skip_pattern]
        elif skip_pattern.__class__.__name__ == "bool":
            skip_pattern = [""]
        elif skip_pattern.__class__.__name__ in ["list", "CommentedSeq"]:
            skip_pattern = skip_pattern

        file_pattern = []
        for pattern in skip_pattern:
            file_pattern.append(pattern + "_" + tag)

        filepaths, duplicates = ul._file_walker(
            dir_path,
            include=file_pattern,
            include_all=False,
            exclude=["pype_config", "attributes"],
            pype_mode=True,
        )

        if len(filepaths) > 0:
            files = []
            for file in filepaths:
                files.append(os.path.basename(file))
            print('\nFound existing files {} - skipped\n'.format((*files,)))
            return True
        else:
            return False

    def _check_final(self):

        ## check components before starting pype to see if something went wrong
        if (
            not hasattr(self.container, "image")
            or self.container.image.__class__.__name__ == "NoneType"
        ):
            raise AttributeError("No image was loaded")
            return
        if (
            not hasattr(self.container, "dir_path")
            or self.container.dir_path.__class__.__name__ == "NoneType"
        ):
            raise AttributeError("Could not determine dir_path to save output.")
            return
        if not hasattr(self, "config") or self.config.__class__.__name__ == "NoneType":
            raise AttributeError(
                "No config file was provided or loading config did not succeed."
            )
            return
        
    def _log(self, lvl, messages, ind=0, passthrough=False):
        
        """Add a message to a logging object.

        Parameters:
        logging (logging.Logger): The logging object to add the message to.
        message (str): The message to be logged.

        """

        if isinstance(messages, str):
            messages_list = messages.split("\n")
        elif isinstance(messages, list):
            messages_list = messages
            
        for message in messages_list:
            if not message == "":
                if message.endswith("\n"):
                    message = message[:-2]
                
                message = ("    " * ind) + message
            
                if lvl == "debug":
                    self.logger.debug(message)
                elif lvl == "info":
                    self.logger.info(message)
                elif lvl == "warning":
                    self.logger.warning(message)
                elif lvl == "error":
                    self.logger.error(message)
                elif lvl == "critical":
                    self.logger.critical(message)
                    
                if self.flags.feedback==False and passthrough==True:
                    hm = datetime.today().strftime("%H:%M:%S")
                    print(hm + ": " + message)


    def _iterate(
        self, 
        config, 
        annotations, 
        execute=True, 
        autoshow=True, 
        interactive=True,
    ):

        flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("execute", bool, execute),
                ("autoshow", bool, autoshow),
                ("interactive", bool, interactive),
            ],
        )

        # =============================================================================
        # FEEDBACK 
        
        if flags.execute and not self.flags.dry_run:
            new_pype_msg = ul._pprint_fill_hbar("| new pype iteration |", ret=True)
            self._log("info", new_pype_msg, 0) 
            
        # =============================================================================

        # reset values
        if not self.flags.dry_run:
            self.container.reset()
        annotation_counter = dict.fromkeys(settings._annotation_types, -1)

        ## apply pype: loop through steps and contained methods
        step_list = self.config["processing_steps"]
        self.config_updated = copy.deepcopy(self.config)
        self.config_parsed_flattened = {}
                
        for step_idx, step in enumerate(step_list):

            # =============================================================================
            # STEP
            # =============================================================================

            if step.__class__.__name__ == "str":
                continue

            ## get step name
            step_name = list(dict(step).keys())[0]
            method_list = list(dict(step).values())[0]
            self.config_parsed_flattened[step_name] = []

            if method_list.__class__.__name__ == "NoneType":
                continue

            ## print current step
            if flags.execute:
                self._log("info", step_name, 0)

            if step_name == "visualization" and flags.execute:

                ## check if canvas is selected, and otherwise execute with default values
                check_list = [
                    list(dict(i).keys())[0] if not isinstance(i, str) else i
                    for i in method_list
                ]
                if (
                    self.container.canvas.__class__.__name__ == "NoneType"
                    and not "select_canvas" in check_list
                ):
                    self.container.run("select_canvas")
                    if flags.interactive and flags.autoshow:
                        self._log("info", "select_canvas (DEFAULT)", 1)
                        
            ## iterate through step list
            for method_idx, method in enumerate(method_list):

                # =============================================================================
                # METHOD / EXTRACTION AND CHECK
                # =============================================================================

                ## format method name and arguments
                if method.__class__.__name__ in ["dict", "ordereddict", "CommentedMap"]:
                    method = dict(method)
                    method_name = list(method.keys())[0]
                    if not list(method.values())[0].__class__.__name__ == "NoneType":
                        method_args = dict(list(method.values())[0])
                    else:
                        method_args = {}
                elif method.__class__.__name__ == "str":
                    method_name = method
                    method_args = {}

                ## feedback
                if flags.execute:
                    self._log("info", method_name, 1)

                ## check if method exists
                if hasattr(eval(step_name), method_name):
                    self.config_parsed_flattened[step_name].append(method_name)
                    pass
                elif self.flags.fix_names:
                    if method_name in settings._legacy_names[step_name]:
                        method_name_updated = settings._legacy_names[step_name][
                            method_name
                        ]
                        self.config_updated["processing_steps"][step_idx][step_name][
                            method_idx
                        ] = {method_name_updated: method_args}
                        method_name = method_name_updated
                        print("Stage: fixed method name")
                else:
                    print("Pype:",
                        "phenopype.{} has no function called {} - will attempt to look for similarly named functions - fix the config file!".format(
                            step_name, method_name
                        )
                    )

                # =============================================================================
                # METHOD / ANNOTATION
                # =============================================================================
                
                if method_name in list(settings._annotation_functions.keys()) + ["convert_annotation"]:
                    if "ANNOTATION" in method_args:
                        annotation_args = dict(method_args["ANNOTATION"])
                        del method_args["ANNOTATION"]
                    else:
                        annotation_args = {}
                        method_args = dict(method_args)
                        self._log("debug", "Pype: Add annotation control args", 0)

                ## annotation params
                if method_name in settings._annotation_functions:

                    annotation_counter[settings._annotation_functions[method_name]] += 1

                    if not "type" in annotation_args:
                        annotation_args.update({"type": settings._annotation_functions[method_name]})
                    if not "id" in annotation_args:
                        annotation_args.update({"id": string.ascii_lowercase[annotation_counter[settings._annotation_functions[method_name]]]})
                    if not "edit" in annotation_args:
                        annotation_args.update({"edit": "overwrite" if method_name in [
                                    "contour_to_mask",
                                    "detect_contour",
                                    "detect_mask",
                                    "compute_shape_features",
                                    "compute_texture_features",
                                    "detect_skeleton",
                                ] else False })


                elif method_name in ["convert_annotation"]:
                    try:
                        annotation_type = method_args["annotation_type_new"]
                        annotation_id = method_args["annotation_id_new"]
                        annotation_args.update({"type":annotation_type, "id":annotation_id, "edit":"overwrite"})
                    except KeyError:
                        self._log("error", "Pype: Missing arguments for annotation conversion", 0)

                else:
                    annotation_args = {}

                ## create ANNOTATION string and add to config
                if annotation_args:
                    annotation_args = ul._yaml_flow_style(annotation_args)
                    method_args_updated = {"ANNOTATION": annotation_args}
                    method_args_updated.update(method_args)
                    self.config_updated["processing_steps"][step_idx][step_name][method_idx] = {method_name: method_args_updated}

                # =============================================================================
                # METHOD / EXECUTE
                # =============================================================================

                ## run method with error handling
                if flags.execute:
                
                    ## open buffer
                    buffer = io.StringIO()
                    
                    ## some function specific switches
                    method_args["interactive"] = flags.interactive                        
                    method_args["zoom_memory"] = self.flags.zoom_memory           
                    method_args["tqdm_off"] = not self.flags.feedback
                    
                    ## excecute and capture stdout
                    try:
                        if not self.flags.debug:
                            with redirect_stdout(buffer):
                                self.container.run(
                                    fun=method_name,
                                    fun_kwargs=method_args,
                                    annotation_kwargs=annotation_args,
                                    annotation_counter=annotation_counter,
                                )
                                stdout = buffer.getvalue()
                                self._log("info", stdout, 2)
                                _config.last_print_msg = ""
                        else:
                            self.container.run(
                                fun=method_name,
                                fun_kwargs=method_args,
                                annotation_kwargs=annotation_args,
                                annotation_counter=annotation_counter,
                            )
                    except Exception as ex:
                        print(buffer.getvalue())
                        error_msg =  f"{step_name}.{method_name}: {str(ex.__class__.__name__)} - {ex}"
                        self._log("error", error_msg, 1)
                        
                        ## cleanup
                        if self.flags.debug:
                            _config.verbose = self.verbose
                            
                            ## raise error and end
                            raise


                    ## check for pype-restart after config change
                    if _config.pype_restart:
                        self._log("debug", "Pype: restart", 0)
                        return
                    
        # sys.stdout = self.old_stdout

        # =============================================================================
        # CONFIG-UPDATE AND AUTOSHOW (optional)
        # =============================================================================

        if not self.config_updated == self.config:
            ul._save_yaml(self.config_updated, self.config_path)
            self._log("debug", "Pype: Updating config; applying staged changes", 0)

        if flags.autoshow and flags.execute:
            try:
                self._log("info", "Pype: AUTOSHOW", 0)
                if self.container.canvas.__class__.__name__ == "NoneType":
                    self.container.run(fun="select_canvas")
                    self._log("info", "select_canvas", 1)
                            
                # =============================================================================
                # FEEDBACK 
        
                closing_msg_list = []
                closing_msg_list.append( ul._pprint_fill_hbar("| finished pype iteration |", ret=True))
                if flags.autoshow:
                    closing_msg_list.append( ul._pprint_fill_hbar("(End with Ctrl+Enter or re-run with Enter)", ret=True))
                self._log("info", closing_msg_list, 0)
                
                # =============================================================================
        
                self.gui = ul._GUI(self.container.canvas, zoom_memory=self.flags.zoom_memory, pype_mode=True)
                self.flags.terminate = self.gui.flags.end_pype
                
            except Exception as ex:
                error_msg =  f"Pype: AUTSHOW {str(ex.__class__.__name__)} - {ex}"
                self._log("error", error_msg, 1)
        else:
            if flags.execute:
                closing_msg = ul._pprint_fill_hbar("| finished pype iteration |", ret=True)
                self._log("info", closing_msg, 0)
                self.flags.terminate = True
