#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

import pickle
import platform
import pprint
import subprocess
import time
import ruamel.yaml

from datetime import datetime
from ruamel.yaml.comments import CommentedMap as ordereddict
from shutil import copyfile, rmtree

from phenopype import presets
from phenopype.settings import *
from phenopype.core import preprocessing, segmentation, measurement, export, visualization
from phenopype.utils import load_image, load_directory, load_image_data, load_meta_data
from phenopype.utils_lowlevel import _image_viewer, _del_rw, _file_walker, _load_pype_config, _create_generic_pype_config
from phenopype.utils_lowlevel import _load_yaml, _show_yaml, _save_yaml, _yaml_file_monitor

#%% settings

pd.options.display.max_rows = 10 # how many rows of pd-dataframe to show
pretty = pprint.PrettyPrinter(width=30) # pretty print short strings
ruamel.yaml.Representer.add_representer(ordereddict, ruamel.yaml.Representer.represent_dict) # suppress !!omap node info


#%% classes

class project: 
    def __init__(self, root, name, **kwargs):
        """
        Initialize a phenopype project with a root directory path and a (folder) name.

        Parameters
        ----------

        root: str
            path to root directory of the project where folder gets created
        name: str
            name of your project and the project folder
        overwrite (optional): bool (default: False)
            overwrite option, if a given root directory already exist 
            (WARNING: also removes all folders inside)
        query(optional: bool (default: False)
            create project without requiring input
        """

        ## kwargs
        flag_overwrite = kwargs.get("overwrite", False)
        flag_query = kwargs.get("query", True)

        ## form dirpath from root-location and name
        root_dir = os.path.join(root, name)

        ## feedback
        print("\n")
        print("--------------------------------------------")
        print("phenopype will create a new project named \"" + name + "\". " +
              "The full path of the project's root directory will be:\n")
        print(root_dir)

        ## decision tree if directory exists
        while True:
            if flag_query == False:
                create = "y"
            else:
                create = input("Proceed? (y/n)\n")
            if create=="y" or create == "yes":
                if os.path.isdir(root_dir):
                    if flag_overwrite == True:
                        rmtree(root_dir, onerror=_del_rw) 
                        print("\n\"" + root_dir + "\" created (overwritten)")
                        pass
                    else:
                        overwrite = input("Warning - project root_dir already exists - overwrite? (y/n)")
                        if overwrite == "y" or overwrite == "yes":
                            rmtree(root_dir, onerror=_del_rw) 
                            print("\n\"" + root_dir + "\" created (overwritten)")
                            pass
                        else:
                            print("\n\"" + root_dir + "\" not created!")
                            print("--------------------------------------------")    
                            break
                else:
                    pass
            else:
                print("\n\"" + root_dir + "\" not created!")
                break

            ## make directories
            self.root_dir = root_dir
            os.makedirs(self.root_dir)
            self.data_dir = os.path.join(self.root_dir, "data")
            os.makedirs(self.data_dir)

            ## lists for files to add
            self.name = name
            self.dirnames = []
            self.dirpaths = []
            self.filenames = []
            self.filepaths = []
            self.fileattr = {}

            ## global project attributes
            project_data = {
                "name": name,
                "date_created": datetime.today().strftime('%Y%m%d_%H%M%S'),
                "date_changed": datetime.today().strftime('%Y%m%d_%H%M%S'),
                "root_dir": self.root_dir,
                "data_dir": self.data_dir}

            _save_yaml(project_data, os.path.join(self.root_dir, "attributes.yaml"))

            print("\nproject attributes written to " + os.path.join(self.root_dir, "attributes.yaml"))
            print("--------------------------------------------")
            break


    def add_files(self, image_dir, **kwargs):
        """Add files to your project from a directory, can look recursively. 
        Specify in- or exclude arguments, filetypes, duplicate-action and copy 
        or link raw files to save memory on the harddrive.
    
        Parameters
        ----------
    
        image_dir: str 
            path to directory with images
        filetypes (optional): list or str
            single or multiple string patterns to target files with certain endings
        include (optional): list or str
            single or multiple string patterns to target certain files to include
        exclude (optional): list or str
            single or multiple string patterns to target certain files to exclude - 
            can overrule "include"
        raw_mode (optional): str (default: "copy")
            how should the raw files be passed on to the phenopype directory tree: 
            "copy" will make a copy of the original file, "link" will only send the 
            link to the original raw file to attributes, but not copy the actual 
            file (useful for big files)
        search_mode (optional): str (default: "dir")
            "dir" searches current directory for valid files; "recursive" walks 
            through all subdirectories
        unique_mode (optional): str (default: "filepath"):
            how to deal with image duplicates - "filepath" is useful if identically 
            named files exist in different subfolders (folder structure will be 
            collapsed and goes into the filename), whereas filename will ignore 
            all similar named files after their first occurrence.
        """
        
        ## kwargs
        flag_overwrite = kwargs.get("overwrite",False)
        flag_raw = kwargs.get("raw_mode", "copy")
        resize = kwargs.get("resize",1)
        search_mode = kwargs.get("search_mode","dir")
        filetypes = kwargs.get("filetypes", [])
        include = kwargs.get("include", [])
        exclude = kwargs.get("exclude", [])
        unique_mode = kwargs.get("unique_mode", "filepath")

        ## collect filepaths
        filepaths, duplicates = _file_walker(image_dir, 
                                             search_mode=search_mode, 
                                             unique_mode=unique_mode, 
                                             filetypes=filetypes, 
                                             exclude=exclude, 
                                             include=include)

        ## loop through files
        for filepath in filepaths:
            
            ## generate phenopype dir-tree
            relpath = os.path.relpath(filepath,image_dir)
            depth = relpath.count("\\")
            relpath_flat = os.path.dirname(relpath).replace("\\","__")
            if depth > 0:
                subfolder_prefix = str(depth) + "__" + relpath_flat + "__"
            else:
                subfolder_prefix = str(depth) + "__" 
            dirname = subfolder_prefix + os.path.splitext(os.path.basename(filepath))[0]
            dirpath = os.path.join(self.root_dir,"data",dirname)

            ## make image-specific directories
            if os.path.isdir(dirpath) and flag_overwrite==False:
                warnings.warn(dirname + " already exists (overwrite=False)")
                continue
            if os.path.isdir(dirpath) and flag_overwrite==True:
                rmtree(dirpath, ignore_errors=True, onerror=_del_rw)
                print("phenopype-project folder " + dirname + " created (overwritten)")
                os.mkdir(dirpath)
            else:
                print("phenopype-project folder " + dirname + " created")
                os.mkdir(dirpath)

            ## copy or link raw files
            if flag_raw == "copy":
                raw_path = os.path.join(dirpath, "raw" + os.path.splitext(os.path.basename(filepath))[1])
                if resize < 1:
                    image = cv2.imread(filepath)
                    image = cv2.resize(image, (0,0), fx=1*resize, fy=1*resize) 
                    cv2.imwrite(raw_path, image)
                else:
                    copyfile(filepath, raw_path)
            elif flag_raw == "link":
                raw_path = filepath

            ## collect attribute-data and save
            image_data = load_image_data(filepath)
            meta_data = load_meta_data(filepath)
            project_data = {
                "project_name": self.name,
                "dirname": dirname,
                "dirpath": dirpath,
                "raw_mode": flag_raw,
                "raw_path": raw_path,
                "resize_factor": resize
                }
            
            attributes = {
                "image": image_data,
                "meta": meta_data,
                "project": project_data}

            ## write attributes file
            _save_yaml(attributes, os.path.join(dirpath, "attributes.yaml"))

            ## add to project object
            if not dirname in self.dirnames:
                self.dirnames.append(dirname)
                self.dirpaths.append(dirpath)
                self.filenames.append(image_data["filename"])
                self.filepaths.append(raw_path)
                self.fileattr[dirname] = attributes

    def add_config(self, name, **kwargs):
        """
        Add pype configuration presets to all project directories. 

        Parameters
        ----------

        name: str
            name of config-file. this gets appended to all files and serves as and
            identifier of a specific analysis pipeline
        preset (optional): str (default: "preset1")
            chose from given presets in phenopype/settings/presets.py 
            (e.g. preset1, preset2, preset3, ...)
        interactive (optional): bool (default: False)
            start a pype and modify preset before saving it to phenopype directories
        overwrite (optional): bool (default: False)
            overwrite option, if a given pype config-file already exist
        """

        ## kwargs
        preset = kwargs.get("preset","preset1")
        flag_interactive = kwargs.get("interactive", None)
        flag_overwrite = kwargs.get("overwrite", False)

        ## modify
        if flag_interactive:
            image_location = os.path.join(self.root_dir,"template_image" + os.path.splitext(self.filenames[0])[1])
            copyfile(self.filepaths[0], image_location)
            config_location = os.path.join(self.root_dir, "pype_config_template-" + name + ".yaml")
            config = _create_generic_pype_config(preset = preset, config_name=name)
            _save_yaml(config, config_location)
            p = pype(image_location, name="template-" + name, config_location=config_location, presetting=True)
            config = p.config
        else:
            config = _load_yaml(eval("presets." + preset))

        ## go through project directories
        for directory in self.dirpaths:
            attr = _load_yaml(os.path.join(directory, "attributes.yaml"))
            pype_preset = {"image": attr["image"],
                           "pype":
                               {"name": name,
                                "preset": preset,
                                "date_created": datetime.today().strftime('%Y%m%d_%H%M%S')}}
            pype_preset.update(config)

            ## save config
            preset_path = os.path.join(directory, "pype_config_" + name + ".yaml")
            dirname = attr["project"]["dirname"]
            if os.path.isfile(preset_path) and flag_overwrite==False:
                print("pype_" + name + ".yaml already exists in " + dirname +  " (overwrite=False)")
                continue
            elif os.path.isfile(preset_path) and flag_overwrite==True:
                print("pype_" + name + ".yaml created for " + dirname + " (overwritten)")
                _save_yaml(pype_preset, preset_path)
            else:
                print("pype_" + name + ".yaml created for " + dirname)
                _save_yaml(pype_preset, preset_path)



class pype:
    def __init__(self, image, name, **kwargs):
        """
        The pype is phenopype’s core method that allows running all functions 
        that are available in the program’s library in sequence. Executing the pype routine 
        will trigger two actions: it will open a yaml configuration file 
        containing instructions for image processing using the default OS text viewer, 
        and a phenopype-window showing the image that was passed on to the pype 
        function as an array, or a character string containing the path to an 
        image on the harddrive (or a directory). Phenopype will parse all functions 
        contained in the config-file in sequence and attempt to apply them to the image 
        (exceptions will be passed, but exceptions returned for diagnostics). 
        The user immediately sees the result and can decide to make changes directly to 
        the opened config-file (e.g. either change function parameters or add new functions), 
        and run the pype again, or to terminate the pype and save all results. 
        The user can store the processed image, any extracted phenotypic information, 
        as well as the modified config-file inside the image directory. 
        By providing unique names, users can store different pype configurations and 
        the associated results side by side. 
        
        Parameters
        ----------

        image: array or str 
            can be either a numpy array or a string that provides the path to source image file 
            or path to a valid phenopype directory
        name: str
            name of pype-config - will be prepended to all results files
        config (optional): str (default: "preset1")
            chose from given presets in phenopype/settings/pype_presets.py (e.g. preset1, preset2, ...)
        interactive (optional): bool (default: False)
            start a pype, modify loaded preset before saving it to phenopype directories
        overwrite (optional): bool (default: False)
            overwrite option, if a given pype config-file already exist
        """
        
        ## pype name check
        if "pype_config_" in name:
            name = name.replace("pype_config_", "")
        elif ".yaml" in name:
            name = name.replace(".yaml", "")
        for char in '[@_!#$%^&*()<>?/\|}{~:]':
            if char in name:
                sys.exit("no special characters allowed in pype name")
        ## kwargs
        flag_show = kwargs.get("show",True)
        flag_skip = kwargs.get("skip", None)
        flag_autoload = kwargs.get("autoload", True)
        flag_autosave = kwargs.get("autosave", True)
        flag_autoshow = kwargs.get("autoshow", True)

        preset = kwargs.get("preset", "preset1")
        config_location = kwargs.get("config_location", None)

        print_settings = kwargs.get("print_settings",False)
        presetting = kwargs.get("presetting", False)
        flag_meta = kwargs.get("meta", True)
        exif_fields = kwargs.get("fields", default_meta_data_fields)
        if not exif_fields.__class__.__name__ == "list":
            exif_fields = [exif_fields]

        ## load image as cointainer from array, file, or directory
        if image.__class__.__name__ == "ndarray":
            self.container = load_image(image, container=True, meta=flag_meta)
            self.container.save_suffix = name
        elif image.__class__.__name__ == "str":
            if os.path.isfile(image):
                self.container = load_image(image, container=True, meta=flag_meta, fields=exif_fields)
                self.container.save_suffix = name
            elif os.path.isdir(image):
                self.container = load_directory(image, meta=flag_meta, fields=exif_fields)
                self.container.save_suffix = name
        else:
            sys.exit("Wrong input - cannot run pype.")

        ## skip directories that already contain specified files
        if flag_skip:
            filepaths, duplicates = _file_walker(self.container.dirpath, 
                                                 include=flag_skip, 
                                                 exclude=["pype_config"], 
                                                 pype_mode=True)
            if len(filepaths)>0:
                print("\nskipped\n")
                return

        ## load config
        if config_location:
            self.config, self.config_location = _load_pype_config(config_location)
        else:
            self.config, self.config_location = _load_pype_config(self.container, config_name=name, preset=preset)

        ## open config file with system viewer
        if flag_show:
            if platform.system() == 'Darwin':       # macOS
                subprocess.call(('open', self.config_location))
            elif platform.system() == 'Windows':    # Windows
                os.startfile(self.config_location)
            else:                                   # linux variants
                subprocess.call(('xdg-open', self.config_location))

        ## initialize
        self.FM = _yaml_file_monitor(self.config_location, print_settings=print_settings)
        update, iv = {}, None
        
        # =============================================================================
        # pype
        # =============================================================================
        
        while True:

            ## get config file and assemble pype
            self.config = copy.deepcopy(self.FM.content)
            if not self.config:
                continue

            ## reiterate
            print("\n\n------------+++ new pype iteration " + 
                  datetime.today().strftime('%Y:%m:%d %H:%M:%S') + 
                  " +++--------------\n\n")
            self.container.reset()
            if flag_autoload:
                self.container.load()
            restart = None
            export_list, show_list = [], []

            ## apply pype
            for step in list(self.config.keys()):
                if step in ["image", "meta", "pype"]:
                    continue
                if not self.config[step]:
                    continue
                if step == "export" and presetting == True:
                    continue
                print(step.upper())
                for item in self.config[step]:
                    try:
                        
                        ## construct method name and arguments
                        if item.__class__.__name__ == "str":
                            method_name = item
                            method_args = {}
                        elif item.__class__.__name__ == "CommentedMap":
                            method_name = list(item)[0]
                            if not list(dict(item).values())[0]:
                                method_args = {}
                            else:
                                method_args = dict(list(dict(item).values())[0])

                        ## collect save-calls
                        if step == "export":
                            export_list.append(method_name)
                            
                        elif step == "visualization":
                            show_list.append(method_name)

                        ## run method
                        print(method_name)
                        method_loaded = eval(step + "." + method_name)
                        restart = method_loaded(self.container, **method_args)
                        
                        ## control
                        if restart:
                            print("RESTART")
                            break
                        
                    except Exception as ex:
                        location = step + "." + method_name + ": " + str(ex.__class__.__name__)
                        print(location + " - " + str(ex))
                        
                if restart:
                    break
            if restart:
                continue

            # save container content, and reset container
            if flag_autoshow:
                self.container.show(show_list=show_list)
            if not presetting:
                if flag_autosave:
                    self.container.save(export_list=export_list)

            ## visualize output
            try:
                if self.container.canvas.__class__.__name__ == "NoneType":
                    self.container.canvas = copy.deepcopy(self.container.image_copy)
                iv = _image_viewer(self.container.canvas, previous=update)
                update = iv.__dict__
            except Exception as ex:
                print("visualisation: " + str(ex.__class__.__name__) + " - " + str(ex))

            ## terminate
            if iv:
                if iv.done:
                    self.FM.stop()
                    print("\n\nTERMINATE")
                    break
            


#%% functions

def save_project(project):
    """
    Save project to root directory

    Parameters
    ----------

    project: phenopype.main.project
        save project file to root dir of project (saves ONLY the python object needed to call file-lists, NOT collected data),
        which needs to be saved separately with the appropriate functions (e.g. "save_csv" and "save_img")
    """
    output_str = os.path.join(project.root_dir, 'project.data')
    with open(output_str, 'wb') as output:
        pickle.dump(project, output, pickle.HIGHEST_PROTOCOL)



def load_project(path):
    """
    Load phenoype project.data file to python namespace

    Parameters
    ----------

    path: path to project.data
        load project file saved to root dir of project
    """
    with open(path, 'rb') as output:
        return pickle.load(output)
