#%% load modules
import copy
import inspect
import os
import pandas as pd
import pickle
import platform
import string
import warnings

import pprint
import subprocess
import sys
import time

from shutil import copyfile

import cv2
import ruamel.yaml
from datetime import datetime
from pathlib import Path
from ruamel.yaml.comments import CommentedMap as ordereddict
from shutil import copyfile, rmtree


from phenopype import __version__ as pp_version
from phenopype.settings import (
    AttrDict,
    confirm_options,
    default_filetypes,
    default_pype_config,
    default_meta_data_fields,
    default_window_size,
    pandas_max_rows,
    pype_config_template_list,
    flag_verbose,
    _annotation_functions,
)
from phenopype.core import preprocessing, segmentation, measurement, export, visualization

from phenopype.utils import load_image, load_pp_directory
from phenopype.utils_lowlevel import (
    _ImageViewer,
    _YamlFileMonitor,
    _del_rw,
    _file_walker,
    _load_image_data,
    _load_pype_config,
    _load_yaml,
    _resize_image,
    _save_yaml,
    _yaml_flow_style,
)

import phenopype._config

#%% settings

pd.options.display.max_rows = pandas_max_rows  # how many rows of pd-dataframe to show
pretty = pprint.PrettyPrinter(width=30)  # pretty print short strings
ruamel.yaml.Representer.add_representer(
    ordereddict, ruamel.yaml.Representer.represent_dict
)  # suppress !!omap node info

#%% classes


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

    def __init__(self, root_dir, load=True, overwrite=False):

        ## set flags
        flags = AttrDict({"overwrite":overwrite, "load":load})    
    
        ## path conversion
        root_dir = root_dir.replace(os.sep, "/")
        root_dir = os.path.abspath(root_dir)
        
        print("--------------------------------------------")
        while True:
            if os.path.isdir(root_dir):
                if all(["attributes.yaml" in os.listdir(root_dir),
                       "data" in os.listdir(root_dir)]):
                    if flags.load and not flags.overwrite:
                        print("Found existing project root directory - loading from:\n" + root_dir)
                        break
                    elif not flags.load and flags.overwrite:
                        pass
                    elif flags.load and flags.overwrite:
                        print("Found existing phenopype project directory at:\n{}\n".format(root_dir))
                        query1 = input("overwrite (y/n)?")
                        if query1 in confirm_options:
                            pass
                        else:
                            print("Aborted - project \"{}\" not overwritten".format(os.path.basename(root_dir)))
                            return
                    rmtree(root_dir, onerror=_del_rw)
                    os.makedirs(root_dir)
                    os.makedirs(os.path.join(root_dir,"data"))
                    print('\n"' + root_dir + '" created (overwritten)')
                    break
                elif len(os.listdir(root_dir)) == 0:
                    print("Found existing project root directory - creating missing directories and files.")
                    os.makedirs(os.path.join(root_dir,"data"))
                    break
                else:
                    print("Directory is neither empty nor a valid phenopype directory - aborting.")
                    
                    return
            else:
                print("Creating a new phenopype project directory at:\n" + root_dir + "\n")
                query2 = input("Proceed? (y/n)\n")
                if query2 in confirm_options:
                    os.makedirs(root_dir)
                    os.makedirs(os.path.join(root_dir,"data"))
                    break
                else:
                    print('\n"' + root_dir + '" not created!')
                    return
    
        ## read directories
        dirnames, dirpaths = os.listdir(os.path.join(root_dir, "data")), []
        for filepath in os.listdir(os.path.join(root_dir, "data")):
            dirpaths.append(os.path.join(root_dir, "data", filepath))

        ## global project attributes
        if not os.path.isfile(os.path.join(root_dir, "attributes.yaml")):
            project_attributes = {
                "project_info": {
                    "date_created": datetime.today().strftime("%Y%m%d%H%M%S"),
                    "date_changed": datetime.today().strftime("%Y%m%d%H%M%S"),
                    "phenopype_version": pp_version,
                    },
                "project_data": None
                }
            _save_yaml(project_attributes, os.path.join(root_dir, "attributes.yaml"))
            print("\nProject \"{}\" successfully created.".format(os.path.basename(root_dir),len(dirpaths)))
        else:
            if len(dirnames) > 0:
                print("\nProject \"{}\" successfully loaded with {} images".format(os.path.basename(root_dir),len(dirpaths)))
            else:
                print("\nProject \"{}\" successfully loaded, but it didn't contain any images!".format(os.path.basename(root_dir)))
                      
        print("--------------------------------------------")
            
        ## attach to instance
        self.root_dir = root_dir
        self.dirnames = dirnames
        self.dirpaths = dirpaths


    def add_files(
        self,
        image_dir,
        filetypes=default_filetypes,
        include=[],
        include_all=True,
        exclude=[],
        mode="copy",
        extension="tif",
        recursive=False,
        overwrite=False,
        resize_factor=1,
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
            "default_filetypes" are configured in settings.py
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
        unique: {"filepath", "filename"}, str, optional:
            how to deal with image duplicates - "filepath" is useful if identically 
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
        extension: {".tif", ".bmp", ".jpg", ".png"}, str, optional
            file extension for "mod" mode
        resize_factor: float, optional
            
        kwargs: 
            developer options
        """

        # kwargs
        ## flags
        flags = AttrDict({"mode": mode, 
                          "recursive": recursive, 
                          "overwrite": overwrite,
                          "resize": False,
                          })

        if resize_factor < 1:
            flags.resize = True
            if not flags.mode=="mod":
                flags.mode = "mod"
                print("Resize factor <1 or >1 - switched to \"mod\" mode")
        
        ## path conversion
        image_dir = image_dir.replace(os.sep, "/")
        image_dir = os.path.abspath(image_dir)
            
        ## collect filepaths
        filepaths, duplicates = _file_walker(
            directory=image_dir,
            recursive=recursive,
            unique=unique,
            filetypes=filetypes,
            exclude=exclude,
            include=include,
            include_all = include_all,
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

        ## loop through files
        for filepath in filepaths:

            ## generate folder paths by flattening nested directories; one 
            ## folder per file
            relpath = os.path.relpath(filepath, image_dir)
            depth = relpath.count("\\")
            relpath_flat = os.path.dirname(relpath).replace("\\", "__")
            if depth > 0:
                subfolder_prefix = str(depth) + "__" + relpath_flat + "__"
            else:
                subfolder_prefix = str(depth) + "__"
                
            dirname = subfolder_prefix + os.path.splitext(os.path.basename(filepath))[0]
            dirpath = os.path.join(self.root_dir, "data", dirname)

            ## make image-specific directories
            if os.path.isdir(dirpath): 
                if flags.overwrite == False:
                    print(
                        "Found image "
                        + relpath
                        + " - "
                        + dirname
                        + " already exists (overwrite=False)"
                    )
                    continue
                elif flags.overwrite == "files":
                    pass
                elif flags.overwrite == "dir":
                    rmtree(dirpath, ignore_errors=True, onerror=_del_rw)
                    print(
                        "Found image "
                        + relpath
                        + " - "
                        + "phenopype-project folder "
                        + dirname
                        + " created (overwrite == \"dir\")"
                    )
                    os.mkdir(dirpath)
            else:
                print(
                    "Found image "
                    + relpath
                    + " - "
                    + "phenopype-project folder "
                    + dirname
                    + " created"
                )
                os.mkdir(dirpath)

            ## load image, image-data, and image-meta-data
            image = load_image(filepath)
            image_name = os.path.basename(filepath)
            image_basename = os.path.splitext(image_name)[0]
            image_data_original = _load_image_data(filepath)
            image_data_phenopype = {
                "date_added": datetime.today().strftime("%Y%m%d%H%M%S"),
                "mode": flags.mode,
                    }

            ## copy or link raw files
            if flags.mode == "copy":
                image_phenopype_path = os.path.join(
                    self.root_dir,
                    "data",
                    dirname,
                    "copy_" + image_name,
                )
                copyfile(filepath, image_phenopype_path)
                image_data_phenopype.update(_load_image_data(image_phenopype_path, path_and_type=False))
                
            elif flags.mode == "mod":
                if resize_factor < 1:
                    image = _resize_image(image, resize_factor)
                if not "." in extension:
                    extension = "." + extension
                image_phenopype_path = os.path.join(
                    self.root_dir,
                    "data",
                    dirname,
                    "mod_" + image_basename + extension,
                )
                if os.path.isfile(image_phenopype_path) and flags.overwrite=="file":
                    print(
                        "Found image "
                        + image_phenopype_path
                        + " in "
                        + dirname
                        + " - overwriting (overwrite == \"files\")"
                    )
                cv2.imwrite(image_phenopype_path, image)
                image_data_phenopype.update({
                    "resize": flags.resize,
                    "resize_factor":resize_factor,
                    })
                image_data_phenopype.update(_load_image_data(image_phenopype_path, path_and_type=False))
                
            elif flags.mode == "link":
                image_phenopype_path = filepath

            ## write attributes file
            attributes = {
                "image_original":image_data_original,
                "image_phenopype":image_data_phenopype}
            if os.path.isfile(os.path.join(dirpath, "attributes.yaml")) and flags.overwrite=="file":
                print("overwriting attributes")
            _save_yaml(
                attributes, os.path.join(dirpath, "attributes.yaml")
            )

        ## list dirs in data and add to project-attributes file in project root
        project_attributes = _load_yaml(os.path.join(self.root_dir, "attributes.yaml"))
        project_attributes["project_data"] = os.listdir(os.path.join(self.root_dir, "data"))
        _save_yaml(
            project_attributes, os.path.join(self.root_dir, "attributes.yaml")
        )
        
        ## add dirlist to project object (always overwrite)
        dirnames = os.listdir(os.path.join(self.root_dir, "data"))
        dirpaths = []
        for dirname in dirnames:
            dirpaths.append(os.path.join(self.root_dir, "data", dirname))
        self.dirnames = dirnames
        self.dirpaths = dirpaths

        print("\nFound {} files".format(len(filepaths)))
        print("--------------------------------------------")
        
        

    def add_config(
        self,
        name,
        template=None,
        interactive=False,
        interactive_image="first",
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

        name: str
            name of config-file. this gets appended to all files and serves as and
            identifier of a specific analysis pipeline
        template: str, optional
            can be either a path to a template or config-file in yaml-format
            or a phenopype template name (e.g. ex1, ex2,...). phenopype comes with 
            a range of templates that correspond to the tutorials and that can 
            be inspected using the following helper functions:
            
            pp.pype_config_template_list        # gives a dictionary with template names
            pp.show_pype_config_template('ex1')  # prints the configuration for example 1

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

        ## kwargs and setup
        flag_interactive = interactive
        flag_overwrite = overwrite
        
        ## for package-testing
        test_params = kwargs.get("test_params", {})
        
        ## load config
        config = _load_pype_config(config=None, template=template, name=name)
        if config.__class__.__name__ == "NoneType":
            return
        
        ## interactive template modification
        if flag_interactive:
            while True:
                if len(self.dirpaths)>0:
                    if interactive_image.__class__.__name__ == "str":
                        if interactive_image == "first":    
                            image_location = self.dirpaths[0]
                            print_msg = "Entered interactive config mode using first image (" + \
                                interactive_image + ")."
                        elif interactive_image in self.dirnames:
                            image_location = os.path.join(
                                self.root_dir,
                                "data",
                                interactive_image,
                            )
                            print_msg = "Entered interactive config mode using " + interactive_image
                        else:
                            print_msg = "Could not enter interactive mode - did not find: " + interactive_image
                            break
                    elif interactive_image.__class__.__name__ == "int":
                        image_location = self.dirpaths[interactive_image-1]
                        print_msg = "Entered interactive config mode using index {}".format(interactive_image) + \
                            " ({})".format(self.dirnames[interactive_image])
                    else:
                        print_msg = "Could not enter interactive mode - wrong input."
                        break
                else:
                    print_msg = "Project contains no images - could not add config files in interactive mode."
                    break
                
                if os.path.isdir(image_location):
                    interactive_container = load_pp_directory(image_location)
                    interactive_container.dirpath = self.root_dir
                else: 
                    print_msg = "Could not enter interactive mode - invalid directory."

                ## save under project root dir
                config_path = os.path.join(
                    self.root_dir, "pype_config_MOD_" + name + ".yaml"
                )
                _save_yaml(config, config_path)
                
                ## run pype 
                p = Pype(
                    interactive_container,
                    name=name,
                    config=config_path,
                    presetting=True,
                    test_params=test_params,
                )
                config = p.config
                break
            print(print_msg)


        ## save config to each directory
        for dirname in self.dirnames:
            
            ## save config
            config_path = os.path.join(
                self.root_dir, 
                "data",
                dirname, 
                "pype_config_" + name + ".yaml"
            )
            if os.path.isfile(config_path) and flag_overwrite == False:
                print(
                    "pype_"
                    + name
                    + ".yaml already exists in "
                    + dirname
                    + " (overwrite=False)"
                )
                continue
            elif os.path.isfile(config_path) and flag_overwrite == True:
                print("pype_" + name + ".yaml created for " + dirname + " (overwritten)")
                config["config_info"]["config_path"] = config_path
                _save_yaml(config, config_path)
            else:
                print("pype_" + name + ".yaml created for " + dirname)
                config["config_info"]["config_path"] = config_path
                _save_yaml(config, config_path)


    def add_reference(self, 
                      name,
                      reference_image, 
                      activate=True,
                      template=False,
                      overwrite=False,
                      **kwargs):
        """
        Add pype configuration presets to all project directories. 

        Parameters
        ----------

        reference_image: str
            name of template image, either project directory or file link. template 
            image gets stored in root directory, and information appended to all 
            attributes files in the project directories
        activate: bool, optional
            writes the setting for the currently active reference to the attributes 
            files of all directories within the project. can be used in conjunction
            with overwrite=False so that the actual reference remains unchanced. this
            setting useful when managing multiple references per project
        overwrite: bool, optional
            overwrite option, if a given pype config-file already exist
        template: bool, optional
            should a template for reference detection be created. with an existing 
            template, phenopype can try to find a reference card in a given image,
            measure its dimensions, and adjust pixel-to-mm-ratio and colour space
        """

        ## kwargs and setup
        flags = AttrDict({"overwrite": overwrite, 
                          "activate": activate})

        reference_name = name
        print_save_msg = "== no msg =="
        
        ## load reference image
        if reference_image.__class__.__name__ == "str":
            if os.path.isfile(reference_image):
                reference_image_path = reference_image
                reference_image = cv2.imread(reference_image_path)
                print("Reference image loaded from " + reference_image_path)
            elif os.path.isdir(reference_image_path):
                reference_image = load_pp_directory(
                   reference_image_path
                )
                reference_image = reference_image.image
                print("Reference image loaded from phenopype dir: \"" + os.path.basename(reference_image_path) + "\"")
            else:
                print("Wrong path - cannot load reference image")
                return
        elif reference_image.__class__.__name__ == "ndarray":
            reference_image_path = "none (array-type)"
            pass
        elif reference_image.__class__.__name__ == "int":
            reference_image_path = os.path.join(self.root_dir, "data", self.dirpaths[reference_image-1])
            reference_image = load_pp_directory(
               reference_image_path
            )
            reference_image = reference_image.image
            print("Reference image loaded from phenopype dir: \"" +  os.path.basename(reference_image_path) + "\"")
        else:
            print("Cannot load reference image - check input")
            return
        
        # =============================================================================
        # METHOD START
        # =============================================================================

        while True:
            
            ## generate reference name and check if exists
            reference_filename = "reference_" + reference_name + ".tif"
            reference_path = os.path.join(self.root_dir, reference_filename)
            
            if os.path.isfile(reference_path) and flags.overwrite == False:
                print_save_msg = "Reference image not saved, file already exists " + \
                 "- use \"overwrite==True\" or chose different name."
                break
            elif os.path.isfile(reference_path) and flags.overwrite == True:
                print_save_msg = "Reference image saved under " + reference_path + " (overwritten)."
                pass
            elif not os.path.isfile(reference_path):
                print_save_msg = "Reference image saved under " + reference_path
                pass
            
            
            ## generate template name and check if exists
            template_name = "reference_template_" + reference_name + ".tif"
            template_path = os.path.join(self.root_dir, template_name)

            if os.path.isfile(template_path) and flags.overwrite == False:
                print_save_msg = "Reference template not saved, file already exists\
                 - use \"overwrite==True\" or chose different name."
                break
            elif os.path.isfile(template_path) and flags.overwrite == True:
                print_save_msg = print_save_msg + "\nReference image saved under " + template_path + " (overwritten)."
                pass
            elif not os.path.isfile(template_path):
                print_save_msg = print_save_msg + "\nReference image saved under " + template_path 
                pass
            
            ## measure reference
            annotation_ref = preprocessing.create_reference(
                reference_image, mask=True
            )

            ## create template from mask coordinates
            coords = annotation_ref["data"]["coord_list"][0]
            template = reference_image[coords[0][1]:coords[2][1], coords[0][0]:coords[1][0]]

            ## create reference attributes
            reference_info = {
                    "date_added":datetime.today().strftime("%Y%m%d%H%M%S"),
                    "reference_image":reference_filename,
                    "original_filepath":reference_image_path,
                    "template_image":template_name,
                    "template_px_mm_ratio": annotation_ref["data"]["px_mm_ratio"],
                    }
                        
            ## load project attributes and temporarily drop project data list to 
            ## be reattched later, so it is always at then end of the file
            reference_dict = {}
            project_attributes = _load_yaml(os.path.join(self.root_dir, "attributes.yaml"))
            if "project_data" in project_attributes:
                project_data = project_attributes["project_data"]
                project_attributes.pop('project_data', None)
            if "reference" in project_attributes:
                reference_dict = project_attributes["reference"]
            reference_dict[reference_name] = reference_info
            
            project_attributes["reference"] = reference_dict
            project_attributes["project_data"] = project_data

            ## save all after successful completion of all method-steps 
            cv2.imwrite(reference_path, reference_image)
            cv2.imwrite(template_path, template)
    
            _save_yaml(project_attributes, os.path.join(self.root_dir, "attributes.yaml"))
            print_save_msg = print_save_msg + "\nSaved reference info to project attributes."
            break
                
        print(print_save_msg)
        
        # =============================================================================
        # METHOD END
        # =============================================================================
        
        ## set active reference information in file specific attributes
        for dirname, dirpath in zip(self.dirnames, self.dirpaths):
            attr = _load_yaml(os.path.join(dirpath, "attributes.yaml"))
            
            ## create nested dict
            if not "reference" in attr:
                attr["reference"] = {}      
            if not "project_level" in attr["reference"]:
                attr["reference"]["project_level"] = {}      
            if not reference_name in attr["reference"]["project_level"]:
                attr["reference"]["project_level"][reference_name] = {}
                
            ## loop through entries and set active reference
            if flags.activate==True:
                for key, value in attr["reference"]["project_level"].items():
                    if key == reference_name:
                        attr["reference"]["project_level"][key]["active"] = True
                    else:
                        attr["reference"]["project_level"][key]["active"] = False
                _save_yaml(attr, os.path.join(dirpath, "attributes.yaml"))
                print("setting active reference to \"" + reference_name + "\" for " + \
                        dirname + " (active=True)")
            else:
                print("could not set active reference for " + dirname + \
                        " (overwrite=False/activate=False)")

    def collect_canvas(self, 
                       name, 
                       folder="canvas", 
                       overwrite=False, 
                       **kwargs):
        
        """
        Collect canvas from each folder in the project tree. Search by 
        name/safe_suffix (e.g. "v1").

        Parameters
        ----------
        name : str
            name of the pype or save_suffix
        folder : str, optional
            folder in the root directory where the results are stored
        overwrite : bool, optional
            should the results be overwritten

        """
        ## kwargs
        flags = AttrDict({"overwrite":overwrite})

        extension = kwargs.get("extension", ".jpg")
        if "." not in extension:
            extension = "." + extension

        results_path = os.path.join(self.root_dir, folder)

        if not os.path.isdir(results_path):
            os.makedirs(results_path)
            print("Created " + results_path)

        ## search string
        search_string = "canvas_" + name + extension

            
        ## append name
        print(search_string)

        ## search
        found, duplicates = _file_walker(
            os.path.join(self.root_dir,"data"),
            recursive=True,
            include=search_string,
            exclude=["pype_config", "attributes", "annotations"],
        )

        ## collect
        for filepath in found:
            print(
                "Collected "
                + os.path.basename(filepath)
                + " from "
                + os.path.basename(os.path.dirname(filepath))
            )
            filename = (
                os.path.basename(os.path.dirname(filepath))
                + "_"
                + os.path.basename(filepath)
            )
            path = os.path.join(results_path, filename)

            ## overwrite check
            while True:
                if os.path.isfile(path) and flags.overwrite == False:
                    print(
                        filename + " not saved - file already exists (overwrite=False)."
                    )
                    break
                elif os.path.isfile(path) and flags.overwrite == True:
                    print(filename + " saved under " + path + " (overwritten).")
                    pass
                elif not os.path.isfile(path):
                    print(filename + " saved under " + path + ".")
                    pass
                copyfile(filepath, path)
                break
            


    def edit_config(
        self,
        name,
        target,
        replacement, 
        **kwargs
    ):
        """
        Add or edit functions in all configuration files of a project. Finds and
        replaces single or multiline string-patterns. Ideally this is done via 
        python docstrings that represent the parts of the yaml file to be replaced.
                
        Parameters
        ----------

        name: str
            name (suffix) of config-file (e.g. "v1" in "pype_config_v1.yaml")
        target: str
            string pattern to be replaced. should be in triple-quotes to be exact
        replacement: str
            string pattern for replacement. should be in triple-quotes to be exact
        """
        
        ## setup
        flag_checked = False
       
        ## go through project directories
        for directory in self.dirpaths:
            dirname = os.path.basename(directory)            

            ## get config path
            config_path = os.path.join(
                self.root_dir, 
                "data",
                dirname, 
                "pype_config_" + name + ".yaml"
            )
            
            ## open config-file
            if os.path.isfile(config_path):
                with open(config_path, "r") as config_text:
                    config_string = config_text.read()
            else:
                print("Did not find config file to edit - check provided name/suffix.")
                return
            ## string replacement
            new_config_string = config_string.replace(target, replacement)
            
            ## show user replacement-result and ask for confirmation
            if flag_checked == False:
                print(new_config_string)
                check = input("This is what the new config may look like (can differ beteeen files) - proceed?")
            
            ## replace for all config files after positive user check
            if check in confirm_options:
                flag_checked = True
                with open(config_path, "w") as config_text:
                    config_text.write(new_config_string)
                
                print("New config saved for " + dirname)
            else:
                print("User check failed - aborting.")
                break 



class Pype(object):
    """
    The pype is phenopype’s core method that allows running all functions 
    that are available in the program’s library in sequence. Users can execute 
    the pype method on a filepath, an array, or a phenopype directory, which 
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
    dirpath: str, optional
        path to an existing directory where all output should be stored
    skip: bool, optional
        skip directories that already have "name" as a suffix in the filename
    feedback: bool, optional
        don't open text editor or window, just apply functions and terminate
    max_dim: int, optional
        maximum dimension that window can have 
    kwargs: 
        developer options
    """

    def __init__(
        self,
        image,
        name,
        config=None,
        template=None,
        dirpath=None,
        skip=False,
        feedback=True,
        save=True,
        **kwargs
    ):

        # =============================================================================
        # CHECKS & INIT
        # =============================================================================
        
        ## kwargs
        global window_max_dim
        window_max_dim = kwargs.get("window_max_dim")
        
        ## flags
        self.flags = AttrDict({"skip": skip, 
                               "feedback": feedback, 
                               "terminate": False,
                               "debug": kwargs.get("debug",False),
                               "save": save})
                                
        ## check name, load container and config
        self._check_pype_name(name=name)
        self._load_container(name=name, image=image, dirpath=dirpath)
        self._load_pype_config(name=name, config=config, template=template)
               
        ## check whether directory is skipped
        if self.flags.skip == True:
            dir_skip = self._check_directory_skip(name=name, dirpath=self.container.dirpath)
            if dir_skip:
                return
            
        ## load existing annotations through container
        self.container.load()

        ## check pype config for annotations
        self._iterate(config=self.config, annotations=self.container.annotations,
                      execute=False, visualize=False, feedback=False)
        time.sleep(1)

        ## final check before starting pype
        self._check_final()
    
        # open config file with system viewer
        if self.flags.feedback:
            self._start_file_monitor()

        ## start log
        self.log = []
        
        # =============================================================================
        # PYPE LOOP   
        # =============================================================================

        ## run pype
        while True:
            
            ## pype restart flag
            phenopype._config.pype_restart = False

            ## refresh config
            self.config = copy.deepcopy(self.YFM.content)
            if not self.config:
                continue
            
            ## run pype config in sequence
            self._iterate(config=self.config, annotations=self.container.annotations)
            
            ## terminate
            if self.flags.terminate:
                self.YFM._stop()
                print("\n\nTERMINATE")         
                break
        
        if self.flags.save and self.flags.terminate:
            if "export" not in self.config_parsed_flattened:
                export_list = []
            else:
                export_list = self.config_parsed_flattened["export"]
            self.container.save(export_list = export_list)
            
            
    def _load_container(self, name, image, dirpath):

        ## load image as cointainer from array, file, or directory
        if image.__class__.__name__ == "ndarray":
            self.container = load_image(path=image, 
                                        load_container=True, 
                                        save_suffix=name)
        elif image.__class__.__name__ == "str":
            if os.path.isfile(image):
                self.container = load_image(path=image, 
                                            load_container=True, 
                                            save_suffix=name)
            elif os.path.isdir(image):
                self.container = load_pp_directory(path=image, 
                                                   dirpath=image, 
                                                   load_container=True, 
                                                   save_suffix=name) 
            else:
                print("Invalid path - cannot run pype.")
                return
        elif image.__class__.__name__ == "Container":
            self.container = image
        else:
            print("Wrong input path or format - cannot run pype.")
            return
    
        ## manually supply dirpath to save files (overwrites container dirpath)
        if not dirpath.__class__.__name__ == "NoneType":
            if not os.path.isdir(dirpath):
                q = input("Save folder {} does not exist - create?.".format(dirpath))
                if q in ["True", "true", "y", "yes"]:
                    os.makedirs(dirpath)
                else:
                    print("Directory not created - aborting")
                    return
            self.container.dirpath = dirpath
           
            
    def _load_pype_config(self, name, config, template):

        ## load pype config (three contexts):
        ## 1) load from existing file
        if all([config.__class__.__name__ == "str",
                template.__class__.__name__ == "NoneType"]):
             self.config = _load_pype_config(config=config, template=None, name=name)
             self.config_path = config       
             if self.config.__class__.__name__ == "NoneType":
                 self.config_path = os.path.join(self.container.dirpath, config)
                 self.config = _load_pype_config(config=self.config_path, template=None, name=name)

             
        ## 2) create new from template or, if already exists, load from file
        if all([config.__class__.__name__ == "NoneType",
                template.__class__.__name__ == "str"]):
            self.config_path = os.path.join(self.container.dirpath,
                                       "pype_config_" + name + ".yaml")
            if os.path.isfile(self.config_path):
                q = input("pype_config_" + name + ".yaml already exists - overwrite?\n" + \
                          "y: yes, file will be overwritten and loaded\n" + 
                          "n: no, existing file will be loaded instead\n" + \
                          "To load an existing file, use \"config\" instead of \"template\".")
                if q in confirm_options:
                    self.config = _load_pype_config(config=None, template=template, name=name)
                    self.config["config_info"]["config_path"] = self.config_path
                    _save_yaml(self.config, self.config_path)
                else: 
                    self.config = _load_pype_config(config=self.config_path, template=None)
            else:
                self.config = _load_pype_config(config=None, template=template, name=name)
                self.config["config_info"]["config_path"] = self.config_path
                _save_yaml(self.config, self.config_path)

                    
        ## 3) check if config file exists in directory (for project directory)
        if all([config.__class__.__name__ == "NoneType",
                template.__class__.__name__ == "NoneType"]):
            config_path = os.path.join(self.container.dirpath,
                                       "pype_config_" + name + ".yaml")
            self.config = _load_pype_config(config=config_path, template=None)
            self.config_path = config_path
            
            
    def _start_file_monitor(self):
        
        if platform.system() == "Darwin":  # macOS
            subprocess.call(("open", self.config_path))
        elif platform.system() == "Windows":  # Windows
            os.startfile(self.config_path)
        else:  # linux variants
            subprocess.call(("xdg-open", self.config_path))

        self.YFM = _YamlFileMonitor(self.config_path)
            
        
    def _check_pype_name(self, name):
        
        ## pype name check
        if "pype_config_" in name:
            name = name.replace("pype_config_", "")
        elif ".yaml" in name:
            name = name.replace(".yaml", "")
        for char in "[@_!#$%^&*()<>?/|}{~:]\\":
            if char in name:
                print("No special characters allowed in pype name - aborting.")
                return
            
            
    def _check_directory_skip(self, name, dirpath):
        ## skip directories that already contain specified files
        
        filepaths, duplicates = _file_walker(
            dirpath,
            include=name,
            exclude=["pype_config"],
            pype_mode=True,
        )
        if len(filepaths) > 0:
            print(
                '\nFound existing result files containing "' + name + '" - skipped\n'
            )
            return True
        else: 
            return False
        
        
    def _check_final(self):
        
        ## check components before starting pype to see if something went wrong
        if (
            not hasattr(self.container, "image")
            or self.container.image.__class__.__name__ == "NoneType"
        ):
            print("Pype error - no image loaded.")
            return
        if (
            not hasattr(self.container, "dirpath")
            or self.container.dirpath.__class__.__name__ == "NoneType"
        ):
            print("Pype error - no dirpath provided.")
            return
        if (
            not hasattr(self, "config")
            or self.config.__class__.__name__ == "NoneType"
        ):
            print("Pype error - no config file provided.")
            return
        
        
    def _iterate(
            self, 
            config,
            annotations,
            execute=True,
            visualize=True,
            feedback=True,
            ):
        
        flags = AttrDict({"execute":execute, "visualize":visualize, "feedback":feedback})

        ## new iteration
        if flags.feedback:
            print(
                "\n\n------------+++ new pype iteration "
                + datetime.today().strftime("%Y:%m:%d %H:%M:%S")
                + " +++--------------\n\n"
            )

        # reset values
        self.container.reset()
        annotation_counter = dict.fromkeys(annotations, -1)

        ## apply pype: loop through steps and contained methods
        step_list = self.config["processing_steps"]
        self.config_updated = copy.deepcopy(self.config)
        self.config_parsed_flattened = {}       
        
        for step_idx, step in enumerate(step_list):
            
            # =============================================================================
            # STEP
            # =============================================================================
                                    
            if step.__class__.__name__=="str":
                continue
            
            ## get step name 
            step_name = list(dict(step).keys())[0]
            method_list = list(dict(step).values())[0]
            self.config_parsed_flattened[step_name] = []
                            
            ## print current step
            if flags.feedback:
                print(step_name.upper())
                

            if step_name == "visualization" and flags.execute:
                
                ## check if canvas is selected, and otherwise execute with default values
                if not "select_canvas" in method_list:
                    print("- autoselect canvas:")
                    self.container.run("select_canvas")
                    
            ## iterate through step list
            for method_idx, method in enumerate(method_list):
                
                # =============================================================================
                # METHOD / EXTRACTION AND CHEC
                # =============================================================================

                ## format method name and arguments       
                if method.__class__.__name__ in ["dict", "ordereddict","CommentedMap"]:
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
                if flags.feedback:
                    print(method_name)
                    
                ## check if method exists
                if hasattr(eval(step_name), method_name):
                    self.config_parsed_flattened[step_name].append(method_name)
                    pass
                else:
                    print("ERROR - {} has no function called {} - please check config file!".format(step_name, method_name))
                    
                    
                # =============================================================================
                # METHOD / ANNOTATION 
                # =============================================================================

                ## annotation params 
                if method_name in _annotation_functions:
                
                    if "ANNOTATION" in method_args:
                        annotation_args = dict(method_args["ANNOTATION"])
                        del method_args["ANNOTATION"]
                    else:
                        annotation_args = {}
                        method_args = dict(method_args)
                        
                        annotation_counter[_annotation_functions[method_name]] += 1
                        if not "type" in annotation_args:
                            annotation_args.update({"type":_annotation_functions[method_name]})
                        if not "id" in annotation_args:
                            annotation_id = string.ascii_lowercase[annotation_counter[_annotation_functions[method_name]]]
                            annotation_args.update({"id":annotation_id})
    
                        annotation_args =  _yaml_flow_style(annotation_args)
                        method_args_updated = {"ANNOTATION":annotation_args}
                        method_args_updated.update(method_args)
                        self.config_updated["processing_steps"][step_idx][step_name][method_idx] = {method_name: method_args_updated}
                else:
                    annotation_args = {}
                    
                # =============================================================================
                # METHOD / EXECUTE  
                # =============================================================================

                ## run method with error handling
                if flags.execute:            
                    try:
                        self.container.run(fun=method_name, 
                                           fun_kwargs=method_args,
                                           annotation_kwargs=annotation_args
                                           )
                    except Exception as ex:
                        if self.flags.debug:
                            raise
                        self.log.append(ex)
                        location = (
                            step_name + "." + method_name + ": " + str(ex.__class__.__name__)
                        )
                        print(location + " - " + str(ex))
                
                ## check for pype-restart after config change
                if phenopype._config.pype_restart:
                    print("BREAK")
                    return
                        
        # =============================================================================
        # CONFIG-UPDATE; FEEDBACK; FINAL VISUALIZATION
        # =============================================================================
                    
        if not self.config_updated == self.config:
            _save_yaml(self.config_updated, self.config_path)
            print("updating pype config file")

        if flags.feedback:
            print(
                "\n\n------------+++ finished pype iteration +++--------------\n" 
                + "-------(End with Ctrl+Enter or re-run with Enter)--------\n\n"
            )
        
        if flags.visualize: 
            try:
                print("AUTOSHOW")
                if self.container.canvas.__class__.__name__ == "NoneType":
                    self.container.run(fun="select_canvas")
                    print("- autoselect canvas")
                    
                self.iv = _ImageViewer(self.container.canvas)
                self.flags.terminate = self.iv.finished
                
            except Exception as ex:
                print(
                    "visualisation: " + str(ex.__class__.__name__) + " - " + str(ex)
                )

