#%% modules
import copy
import inspect
import os
import pandas as pd
import pickle
import platform
import pprint
import subprocess
import sys

import cv2
import ruamel.yaml
from datetime import datetime
from ruamel.yaml.comments import CommentedMap as ordereddict
from shutil import copyfile, rmtree

from phenopype import __version__ as pp_version
from phenopype.settings import (
    confirm_options,
    default_filetypes,
    default_pype_config,
    default_meta_data_fields,
    pandas_max_rows,
    template_dir,
    template_list,
)
from phenopype.core import preprocessing, segmentation, measurement, export, visualization

from phenopype.core.preprocessing import resize_image
from phenopype.utils import load_image, load_directory, load_image_data, load_meta_data
from phenopype.utils_lowlevel import (
    _image_viewer,
    _del_rw,
    _file_walker,
    _load_pype_config,
    _create_generic_pype_config,
)
from phenopype.utils_lowlevel import (
    _load_yaml,
    _show_yaml,
    _save_yaml,
    _yaml_file_monitor,
)

#%% settings

pd.options.display.max_rows = pandas_max_rows  # how many rows of pd-dataframe to show
pretty = pprint.PrettyPrinter(width=30)  # pretty print short strings
ruamel.yaml.Representer.add_representer(
    ordereddict, ruamel.yaml.Representer.represent_dict
)  # suppress !!omap node info


#%% classes


class project:
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
        phenopype project object
    """

    def __init__(self, root_dir, overwrite=False):

        ## kwargs
        flag_overwrite = overwrite

        ## path conversion
        root_dir = root_dir.replace(os.sep, "/")
        root_dir = os.path.abspath(root_dir)

        ## feedback
        print("--------------------------------------------")
        print("Phenopype will create a new project at\n" + root_dir + "\n")

        ## decision tree if directory exists
        while True:
            create = input("Proceed? (y/n)\n")
            if create == "y" or create == "yes":
                if os.path.isdir(root_dir):
                    if flag_overwrite == True:
                        rmtree(root_dir, onerror=_del_rw)
                        print('\n"' + root_dir + '" created (overwritten)')
                        pass
                    else:
                        overwrite = input(
                            "Warning - project root_dir already exists - overwrite? (y/n)"
                        )
                        if overwrite == "y" or overwrite == "yes":
                            rmtree(root_dir, onerror=_del_rw)
                            print('\n"' + root_dir + '" created (overwritten)')
                            pass
                        else:
                            print('\n"' + root_dir + '" not created!')
                            print("--------------------------------------------")
                            break
                else:
                    pass
            else:
                print('\n"' + root_dir + '" not created!')
                break

            ## make directories
            self.root_dir = root_dir
            os.makedirs(self.root_dir)
            os.makedirs(os.path.join(self.root_dir,"data"))

            ## global project attributes
            project_info = {
                "date_created": datetime.today().strftime("%Y%m%d%H%M%S"),
                "date_changed": datetime.today().strftime("%Y%m%d%H%M%S"),
                "phenopype_version": pp_version,
            }

            project_attributes = {
                "info":project_info,
                "data":None}

            _save_yaml(project_attributes, os.path.join(self.root_dir, "attributes.yaml"))

            print(
                "\nproject attributes written to "
                + os.path.join(self.root_dir, "attributes.yaml")
            )
            print("--------------------------------------------")
            break

    def add_files(
        self,
        image_dir,
        filetypes=default_filetypes,
        include=[],
        include_all=True,
        exclude=[],
        mode="copy",
        recursive=False,
        overwrite=False,
        resize=1,
        unique="path",
        **kwargs
    ):
        """
        Add files to your project from a directory, can look recursively. 
        Specify in- or exclude arguments, filetypes, duplicate-action and copy 
        or link raw files to save memory on the harddrive. For each found image,
        a folder will be created in the "data" folder within the projects root
        directory. If found images are in subfolders and search_mode is 
        recursive, the respective phenopype directories will be created with 
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
        mode: {"copy", "link"} str, optional
            how should the raw files be passed on to the phenopype directory tree: 
            "copy" will make a copy of the original file, "link" will only send the 
            link to the original raw file to attributes, but not copy the actual 
            file (useful for big files)
        recursive: (optional): bool,
            "False" searches only current directory for valid files; "True" walks 
            through all subdirectories
        unique: {"filepath", "filename"}, str, optional:
            how to deal with image duplicates - "filepath" is useful if identically 
            named files exist in different subfolders (folder structure will be 
            collapsed and goes into the filename), whereas filename will ignore 
            all similar named files after their first occurrence.
        kwargs: 
            developer options
        """

        # kwargs
        flag_mode = mode
        flag_overwrite = overwrite
        flag_resize = False
        if resize < 1:
            flag_resize = True
        resize_factor = resize
        
        ## path conversion
        image_dir = image_dir.replace(os.sep, "/")
        image_dir = os.path.abspath(image_dir)
            
        ## load_project attributes file from root folder
        project_attributes = _load_yaml(os.path.join(self.root_dir, "attributes.yaml"))

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
            + str(mode)
            + ", recursive: "
            + str(recursive)
            + ", resize: "
            + str(flag_resize)
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
            if os.path.isdir(dirpath) and flag_overwrite == False:
                print(
                    "Found image "
                    + relpath
                    + " - "
                    + dirname
                    + " already exists (overwrite=False)"
                )
                continue
            if os.path.isdir(dirpath) and flag_overwrite == True:
                rmtree(dirpath, ignore_errors=True, onerror=_del_rw)
                print(
                    "Found image "
                    + relpath
                    + " - "
                    + "phenopype-project folder "
                    + dirname
                    + " created (overwritten)"
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
            image_basename = os.path.basename(filepath)
            image_data_original = load_image_data(filepath)
            image_data_phenopype = {
                "mode": flag_mode,
                "date_added": datetime.today().strftime("%Y%m%d%H%M%S"),
                    }

            ## copy or link raw files
            if flag_mode == "copy":
                image_phenopype_path = os.path.join(
                    self.root_dir,
                    "data",
                    dirname,
                    "copy_" + image_basename,
                )
                if resize_factor < 1:
                    image = resize_image(image, resize_factor)
                cv2.imwrite(image_phenopype_path, image)
                image_data_phenopype.update({"resize": flag_resize})
                image_data_phenopype.update(load_image_data(image_phenopype_path))

            elif flag_mode == "link":
                if resize_factor < 1:
                    print("cannot resize image in link mode")
                image_phenopype_path = filepath

            ## write attributes file
            attributes = {
                "image_original":image_data_original,
                "image_phenopype":image_data_phenopype}
            
            _save_yaml(
                attributes, os.path.join(dirpath, "attributes.yaml")
            )

            
        ## list dirs in data and add to project-attributes file in project root
        project_attributes["data"] = os.listdir(os.path.join(self.root_dir, "data"))
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
        overwrite=False,
        idx=0,
        typ="rt",
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
        preset: str, optional
            can be either a string denoting a template name (e.g. preset1, preset2, 
            landamarking1, ... - in "phenopype/settings/presets.py") or a path to a 
            compatible yaml file
        interactive: bool, optional
            start a pype and modify preset before saving it to phenopype directories
        overwrite: bool, optional
            overwrite option, if a given pype config-file already exist
        kwargs: 
            developer options
        """

        ## kwargs and setup
        flag_interactive = interactive
        flag_overwrite = overwrite
        step_names = ["preprocessing", 
                      "segmentation", 
                      "measurement",  
                      "visualization", 
                      "export"]
        
        ## load config
        if template.__class__.__name__ == "str":
            if os.path.isfile(template):
                template_name = os.path.basename(template)
                template_path = template
                user_config = _load_yaml(template, typ=typ)
                print("Custom user config template loaded from \"{}\"".format(template))
                if user_config.__class__.__name__ in ["dict", 'CommentedMap']:
                    if "info" in user_config:
                        user_config.pop('info', None)
                        print("Removed existing \"info\" section")
                    if "steps" in user_config:
                        config_steps = user_config["steps"]
                    else:
                        print("Check for correct template structure")
                        return
                elif user_config.__class__.__name__ in ["list",'CommentedSeq'] and any(step in user_config[0] for step in step_names):
                    config_steps = user_config
            else:
                if not template.endswith(".yaml"):
                    template_name =  template + ".yaml" 
                else:
                    template_name = template
                if template_name in template_list:
                    template_path = os.path.join(template_dir, template_name)
                    config_steps = _load_yaml(template_path)
                    print("Phenopype template {} loaded".format(template))
                else:
                    print("Template not found")
                    return
        elif template.__class__.__name__ == "NoneType":
            print("No template provided")
            return
                
        config_info = {"config_name":name,
         "template_name":template_name,
         "template_path":template_path,
         "date_created":datetime.today().strftime("%Y%m%d%H%M%S"),
         "date_last_modified":None}


        config = {"info":config_info,
                  "steps":config_steps}

        ## interactive template modification
        if flag_interactive:
            image_location = os.path.join(
                self.root_dir,
                "pype_template_image" + os.path.splitext(self.filenames[idx])[1],
            )
            
            if os.path.isfile(self.filepaths[idx]):
                template_origin = self.filepaths[idx]
            else: 
                template_origin = os.path.join(self.dirpaths[idx], "raw" + os.path.splitext(os.path.basename(self.filepaths[idx]))[1])
             
            copyfile(template_origin, image_location)
            config_location = os.path.join(
                self.root_dir, "pype_config_template-" + name + ".yaml"
            )
            _save_yaml(config, config_location, typ="safe")
            p = pype(
                image_location,
                name="template-" + name,
                config_location=config_location,
                presetting=True,
            )
            config = p.config

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
                _save_yaml(config, config_path, typ=typ)
            else:
                print("pype_" + name + ".yaml created for " + dirname)
                _save_yaml(config, config_path, typ=typ)



    def add_scale(self, reference_image, overwrite=False, **kwargs):
        """
        Add pype configuration presets to all project directories. 

        Parameters
        ----------

        reference_image: str
            name of template image, either project directory or file link. template 
            image gets stored in root directory, and information appended to all 
            attributes files in the project directories
        overwrite: bool, optional
            overwrite option, if a given pype config-file already exist
        template: bool, optional
            should a template for scale detection be created. with an existing 
            template, phenopype can try to find a reference card in a given image,
            measure its dimensions, and adjust pixel-to-mm-ratio and colour space
        """

        ## kwargs
        flag_overwrite = overwrite
        test_params = kwargs.get("test_params", {})

        ## load template image
        if reference_image.__class__.__name__ == "str":
            if os.path.isfile(reference_image):
                reference_image = cv2.imread(reference_image)
            elif os.path.isdir(os.path.join(self.data_dir, reference_image)):
                attr = _load_yaml(
                    os.path.join(self.data_dir, reference_image, "attributes.yaml")
                )
                reference_image = cv2.imread(
                    os.path.join(self.root_dir, attr["project"]["raw_path"])
                )
            elif reference_image in self.dirnames:
                attr = _load_yaml(
                    os.path.join(self.data_dir, reference_image, "attributes.yaml")
                )
                reference_image = cv2.imread(attr["project"]["raw_path"])
            else:
                print("wrong path - cannot load reference image")
                return
        elif reference_image.__class__.__name__ == "ndarray":
            pass
        elif reference_image.__class__.__name__ == "int":
            reference_image = cv2.imread(self.filepaths[reference_image])
        else:
            print("wrong type - cannot load reference image")
            return

        ## save template
        template_path = os.path.join(self.root_dir, "scale_template.jpg")
        while True:
            if os.path.isfile(template_path) and flag_overwrite == False:
                print(
                    "- scale template not saved - file already exists (overwrite=False)."
                )
                break
            elif os.path.isfile(template_path) and flag_overwrite == True:
                print("- scale template saved under " + template_path + " (overwritten).")
                pass
            elif not os.path.isfile(template_path):
                print("- scale template saved under " + template_path + ".")
                pass

            ## measure scale
            px_mm_ratio, df_masks, template = preprocessing.create_scale(
                reference_image, template=True, test_params=test_params
            )
            cv2.imwrite(template_path, template)
            break

        ## save scale information
        for directory in self.dirpaths:
            attr = _load_yaml(os.path.join(self.root_dir, directory, "attributes.yaml"))
            if not "scale" in attr:
                print("added scale information to " + attr["project"]["dirname"])
                pass
            elif "scale" in attr and flag_overwrite:
                print(
                    "added scale information to "
                    + attr["project"]["dirname"]
                    + " (overwritten)"
                )
                pass
            elif "scale" in attr and not flag_overwrite:
                print(
                    "could not add scale information to "
                    + attr["project"]["dirname"]
                    + " (overwrite=False)"
                )
                continue
            attr["scale"] = {
                "template_px_mm_ratio": px_mm_ratio,
                "template_path": template_path,
            }
            _save_yaml(attr, os.path.join(self.root_dir, directory, "attributes.yaml"))

    def collect_results(self, name, files=None, folder="results", overwrite=False):
        """
        Collect results (images or CSVs) from the entire data folder. Search by 
        pype name (e.g. "v1") and filter by filetypes (e.g. landmarks, 
        contours or colours)

        Parameters
        ----------
        name : str
            name of the pype or save_suffix
        files : str or list, optional
            filetypes to look for (e.g. landmarks, contours or colours)
        folder : str, optional
            folder in the root directory where the results are stored
        overwrite : bool, optional
            should the results be overwritten

        """
        ## kwargs
        flag_overwrite = overwrite

        results_path = os.path.join(self.root_dir, folder)

        if not os.path.isdir(results_path):
            os.makedirs(results_path)
            print("Created " + results_path)

        ## search string
        if not files.__class__.__name__ == "NoneType":
            if not files.__class__.__name__ == "list":
                files = [files]
            search_strings = []
            for file in files:
                if not name == "":
                    search_strings.append(file + "_" + name)
                else:
                    search_strings.append(file)
        else:
            search_strings = name

        ## search
        found, duplicates = _file_walker(
            self.data_dir,
            search_mode="recursive",
            include=search_strings,
            exclude=["pype_config"],
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
                if os.path.isfile(path) and flag_overwrite == False:
                    print(
                        filename + " not saved - file already exists (overwrite=False)."
                    )
                    break
                elif os.path.isfile(path) and flag_overwrite == True:
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
        replaces single or multiline string-patterns 
        

        Parameters
        ----------

        name: str
            name (suffix) of config-file (e.g. "v1" in "pype_config_v1.yaml")
        target: str
            string pattern to be replaced. should be multiline to be exact
        replacement: str
            string pattern for replacement. should be multiline to be exact
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


                
    @staticmethod
    def save(project, overwrite=False):
        """
        Save project to root directory
    
        Parameters
        ----------
    
        project: phenopype.main.project
            save project file to root dir of project (saves ONLY the python object 
            needed to call file-lists, NOT collected data), which needs to be saved 
            separately with the appropriate export functions (e.g. 
            :func:`phenopype.export.save_contours` or :func:`phenopype.export.save_canvas`)
        overwrite: bool, optional
            should existing project data be overwritten
        """
        ## kwargs
        flag_overwrite = overwrite

        ## save project
        output_path = os.path.join(project.root_dir, "project.data")
        while True:
            if os.path.isfile(output_path) and flag_overwrite == False:
                print("Project data not saved - file already exists (overwrite=False).")
                break
            elif os.path.isfile(output_path) and flag_overwrite == True:
                print("Project data saved under " + output_path + " (overwritten).")
                pass
            elif not os.path.isfile(output_path):
                print("Project data saved under " + output_path + ".")
                pass
            with open(output_path, "wb") as output:
                pickle.dump(project, output, pickle.HIGHEST_PROTOCOL)
            break

    @staticmethod
    def load(root_dir):
        """
        Load phenoype project.data file to Python namespace.
    
        Parameters
        ----------
        root_dir: str
            path to project root directory that containes "project.data"
            and "attributes.yaml"

        Returns
        -------
        project: project
            phenopype project object
        """
        
        ## path conversion
        root_dir = root_dir.replace(os.sep, "/")
        root_dir = os.path.abspath(root_dir)

        ## load pickled project object
        if "attributes.yaml" in os.listdir(root_dir) and \
            "project.data" in os.listdir(root_dir):
            project_data_path = os.path.join(root_dir, "project.data")
            with open(project_data_path, "rb") as output:
                proj = pickle.load(output)
            proj.root_dir = root_dir
                    
            ## add dirlist to project object (always overwrite)
            dirnames = os.listdir(os.path.join(proj.root_dir, "data"))
            dirpaths = []
            for dirname in dirnames:
                dirpaths.append(os.path.join(proj.root_dir, "data", dirname))
            proj.dirnames = dirnames
            proj.dirpaths = dirpaths
            
            print("--------------------------------------------")
            print("Project loaded from \n" + proj.root_dir)
            print("\ntogether with {} project image files".format(len(proj.dirpaths)))
            print("--------------------------------------------")
            
            return proj
                
        else:
            print("Could not load phenopype project - no \"attributes.yaml\" \
                  or \"project.data\" found in " + root_dir)



class pype:
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
    config_preset: str, optional
        chose from given presets in phenopype/settings/pype_presets.py 
        (e.g. preset1, preset2, ...)
    config_location: str, optional
        custom path to a pype template (needs to adhere to yaml syntax and 
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
        config_preset=None,
        config_location=None,
        dirpath=None,
        skip=False,
        feedback=True,
        delay=100,
        max_dim=1000,
        overwite=True,
        **kwargs
    ):

        ## legacy
        preset = kwargs.get("preset")
        if (
            config_preset.__class__.__name__ == "NoneType"
            and not preset.__class__.__name__ == "NoneType"
        ):
            config_preset = preset

        ## pype name check
        if "pype_config_" in name:
            name = name.replace("pype_config_", "")
        elif ".yaml" in name:
            name = name.replace(".yaml", "")
        for char in "[@_!#$%^&*()<>?/|}{~:]\\":
            if char in name:
                sys.exit("no special characters allowed in pype name")

        ## kwargs
        flag_feedback = feedback
        flag_skip = skip
        flag_autoload = kwargs.get("autoload", True)
        flag_autosave = kwargs.get("autosave", True)
        flag_autoshow = kwargs.get("autoshow", False)
        flag_presetting = kwargs.get("presetting", False)
        flag_meta = kwargs.get("meta", True)
        flag_overwrite = kwargs.get("overwrite", True) 
        exif_fields = kwargs.get("fields", default_meta_data_fields)
        if not exif_fields.__class__.__name__ == "list":
            exif_fields = [exif_fields]
        test_params = kwargs.get("test_params", None)
        if test_params.__class__.__name__ == "dict":
            flag_test_mode = True
        else:
            flag_test_mode = False

        ## load image as cointainer from array, file, or directory
        if image.__class__.__name__ == "ndarray":
            self.container = load_image(image, cont=True, meta=flag_meta)
            self.container.save_suffix = name
        elif image.__class__.__name__ == "str":
            if os.path.isfile(image):
                self.container = load_image(image, cont=True, meta=False)
                self.container.save_suffix = name
            elif os.path.isdir(image):
                self.container = load_directory(image, meta=flag_meta, fields=exif_fields)
                self.container.save_suffix = name
            else:
                sys.exit("Invalid path - cannot run pype.")
        else:
            sys.exit("Wrong input format - cannot run pype.")

        ## emergency check
        if (
            not hasattr(self.container, "image")
            or self.container.image.__class__.__name__ == "NoneType"
        ):
            sys.exit("Internal error - no image loaded.")

        ## supply dirpath manually
        if not dirpath.__class__.__name__ == "NoneType":
            if not os.path.isdir(dirpath):
                q = input("Save folder {} does not exist - create?.".format(dirpath))
                if q in ["True", "true", "y", "yes"]:
                    os.makedirs(dirpath)
                else:
                    print("Directory not created - aborting")
                    return
            self.container.dirpath = dirpath

        ## skip directories that already contain specified files
        if flag_skip == True:
            filepaths, duplicates = _file_walker(
                self.container.dirpath,
                include=name,
                exclude=["pype_config"],
                pype_mode=True,
            )
            if len(filepaths) > 0:
                print(
                    '\n found existing result files containing "' + name + '" - skipped\n'
                )
                return

        ## load config
        if config_location:
            self.config, self.config_location = _load_pype_config(config_location)
        else:
            self.config, self.config_location = _load_pype_config(
                self.container, config_name=name, preset=config_preset
            )

        ## open config file with system viewer
        if flag_feedback and not flag_test_mode:
            if platform.system() == "Darwin":  # macOS
                subprocess.call(("open", self.config_location))
            elif platform.system() == "Windows":  # Windows
                os.startfile(self.config_location)
            else:  # linux variants
                subprocess.call(("xdg-open", self.config_location))

        ## initialize
        self.FM = _yaml_file_monitor(self.config_location, delay)
        update, terminate, self.iv = {}, False, None

        # =============================================================================
        # pype
        # =============================================================================

        while True:

            ## get config file and assemble pype
            self.config = copy.deepcopy(self.FM.content)
            if not self.config:
                continue

            ## new iteration
            print(
                "\n\n------------+++ new pype iteration "
                + datetime.today().strftime("%Y:%m:%d %H:%M:%S")
                + " +++--------------\n\n"
            )

            # reset values
            self.container.reset()
            if flag_autoload and not flag_presetting:
                self.container.load()
            restart = None
            export_list, show_list = [], []

            ## apply pype: loop through steps and contained methods
            for step in self.config:
                step_name = step[0]
                step_method_list = step[1]
                      
                ## skip cases: meta-steps, none, or presetting mode
                if step_name in ["image", "meta", "pype"]:
                    continue
                if step_method_list.__class__.__name__=="NoneType":
                    continue
                if step_name == "export" and flag_presetting == True:
                    continue
                
                ## feedback: print step name
                print(step_name.upper())
                
                ## iterate through step list
                for method in step_method_list:
                    
                    ## re-format method if necessary
                    if method.__class__.__name__ == "str":
                        method = [(method,{})]
                        
                    ## format method name and arguments
                    method_name = method[0][0]
                    if method[0][1].__class__.__name__ == "NoneType" :
                        method_arguments = {}
                    else:
                        method_arguments = dict(method[0][1])
                        
                    if not flag_feedback:
                        if method_name in ["create_mask"]:
                            continue
                        elif method_name == "draw":
                            method_arguments.update({"mode": "silent"})

                    ## collect save- and visualization calls
                    if step_name == "export":
                        export_list.append(method_name)
                    elif step_name == "visualization":
                        show_list.append(method_name)
                        
                    ## check if canvas is selcted, and otherwise execute with default values
                    if (
                        not "select_canvas" in show_list
                        and self.container.canvas.__class__.__name__ == "NoneType"
                    ):
                        visualization.select_canvas(self.container)
                        print("- autoselect canvas")

                    ## check if method has max_dim option and otherwise add
                    args = inspect.getfullargspec(eval(step_name + "." + method_name)).args
                    if "max_dim" in args and not "max_dim" in method_arguments:
                        method_arguments["max_dim"] = max_dim
                        
                    try:
                        ## run method
                        print(method_name)
                        method_loaded = eval(step_name + "." + method_name)
                        restart = method_loaded(self.container, **method_arguments)
    
                        ## control
                        if restart:
                            print("RESTART")
                            break

                    except Exception as ex:
                        location = (
                            step_name + "." + method_name + ": " + str(ex.__class__.__name__)
                        )
                        print(location + " - " + str(ex))

                if restart:
                    break
            if restart:
                continue

            # save container
            if flag_autoshow:
                self.container.show(show_list=show_list)
            if not flag_presetting:
                if flag_autosave:
                    self.container.save(export_list=export_list, overwrite=flag_overwrite)

            ## visualize output
            if flag_feedback:
                try:
                    if self.container.canvas.__class__.__name__ == "NoneType":
                        visualization.select_canvas(self.container)
                        print("- autoselect canvas")
                    if flag_test_mode:
                        update = test_params
                    self.iv = _image_viewer(self.container.canvas, previous=update, max_dim=max_dim)
                    update, done, terminate = self.iv.__dict__, self.iv.done, self.iv.finished
                except Exception as ex:
                    print(
                        "visualisation: " + str(ex.__class__.__name__) + " - " + str(ex)
                    )

            ## terminate
            if terminate or not flag_feedback:
                self.FM.stop()
                print("\n\nTERMINATE")
                break
