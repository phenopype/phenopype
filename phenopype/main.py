#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

import pickle
import platform
import pprint
import subprocess
import ruamel.yaml

from datetime import datetime
from ruamel.yaml.comments import CommentedMap as ordereddict
from shutil import copyfile, rmtree

from phenopype.utils import load_image, load_directory, load_image_data, load_meta_data
from phenopype.utils_lowlevel import _image_viewer, _del_rw, _file_walker, _load_pype_config, _generic_pype_config
from phenopype.utils_lowlevel import _load_yaml, _show_yaml, _save_yaml, _yaml_file_monitor
from phenopype.settings import presets, colours
from phenopype.core import preprocessing, segmentation, measurement, export, visualization

#%% settings

pd.options.display.max_rows = 10 # how many rows of pd-dataframe to show
pretty = pprint.PrettyPrinter(width=30) # pretty print short strings
ruamel.yaml.Representer.add_representer(ordereddict, ruamel.yaml.Representer.represent_dict) # suppress !!omap node info


#%% classes

class project: 
    def __init__(self, root, name, **kwargs):
        """
        Initialize a phenopype project with a root directory and a name.
        
        Parameters
        ----------
    
        root_dir: str
            root directory of the project
        name: str
            name of your project
        overwrite (optional): bool (default: False)
            overwrite option, if a given root directory already exist
        ask (optional): bool (default: False)
            perform actions without asking for input
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
                        rmtree(root_dir, ignore_errors=True, onerror=_del_rw) 
                        print("\n\"" + root_dir + "\" created (overwritten)")
                        pass
                    else:
                        overwrite = input("Warning - project root_dir already exists - overwrite? (y/n)")
                        if overwrite == "y" or overwrite == "yes":
                            rmtree(root_dir, ignore_errors=True, onerror=_del_rw) 
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
            self.dirlist = []
            self.rawlist = []
            self.files = {}

            ## global project attributes
            project_data = {
                "name": name,
                "date_created": datetime.today().strftime('%Y%m%d_%H%M%S'),
                "date_changed": datetime.today().strftime('%Y%m%d_%H%M%S')}
            project_io = {
                "root_dir": self.root_dir,
                "data_dir": self.data_dir}
            project_attributes = ordereddict([('information', 
                    [ordereddict([('project_data', project_data)]), 
                     ordereddict([('project_io', project_io)])
                     ])])
            _save_yaml(project_attributes, os.path.join(self.root_dir, "attributes.yaml"))

            print("\nproject attributes written to " + os.path.join(self.root_dir, "attributes.yaml"))
            print("--------------------------------------------")
            break


    def add_files(self, image_dir, **kwargs):
        """Add files to your project from a directory, can look recursively. 
        Optional: specify a search string for filenames or file extensions. 
    
        Parameters
        ----------
    
        image_dir: str 
            path to directory with images                             

        raw_mode (optional): str (default: "copy")
            how should the raw files be passed on to the phenopype directory tree: "copy" will make a copy of the original file, 
            "link" will only send the link to the original raw file to attributes, but not copy the actual file (useful for big files)
            
        search_mode (optional): str (default: "dir")
            "dir" searches current directory for valid files; "recursive" walks through all subdirectories
        filetypes (optional): list of str
            single or multiple string patterns to target files with certain endings
        include (optional): list of str
            single or multiple string patterns to target certain files to include
        exclude (optional): list of str
            single or multiple string patterns to target certain files to exclude - can overrule "include"
        unique_by (optional): str (default: "filepath")
            how should unique files be identified: "filepath" or "filename". "filepath" is useful, for example, 
            if identically named files exist in different subfolders (folder structure will be collapsed and goes into the filename),
            whereas filename will ignore all those files after their first occurrence.
        """
        
        ## kwargs
        flag_overwrite = kwargs.get("overwrite",False)
        flag_raw = kwargs.get("raw_mode", "copy")
        resize = kwargs.get("resize",1)
        search_mode = kwargs.get("search_mode","dir")
        filetypes = kwargs.get("filetypes", [])
        include = kwargs.get("include", [])
        exclude = kwargs.get("exclude", [])
        unique_by = kwargs.get("unique_by", "filepath")

        ## collect filepaths
        filepaths, duplicates = _file_walker(image_dir, search_mode=search_mode, unique_by=unique_by, filetypes=filetypes, exclude=exclude, include=include)


        ## loop through files
        for filepath in filepaths:

            ## flatten folder structure
            relpath = os.path.relpath(filepath,image_dir)
            depth = relpath.count("\\")
            relpath_flat = os.path.dirname(relpath).replace("\\","__")
            if depth > 0:
                subfolder_prefix = str(depth) + "__" + relpath_flat + "__"
            else:
                subfolder_prefix = str(depth) + "__" 
                
            ## image paths
            dirname = subfolder_prefix + os.path.splitext(os.path.basename(filepath))[0]
            dirpath = os.path.join(self.root_dir,"data",dirname)

            ## make image-specific directories
            if os.path.isdir(dirpath) and flag_overwrite==False:
                print(dirname + " already exists (overwrite=False)")
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
            self.dirnames.append(dirname)
            self.dirlist.append(dirpath)
            self.rawlist.append(raw_path)
            self.files[dirname] = attributes

    def add_config(self,  **kwargs):
        """
        Add pype configuration presets to all project directories

        Parameters
        ----------

        name (optional): str (default: "v1")
            name of config-file
        preset (optional): str (default: "preset1")
            chose from given presets in phenopype/settings/pype_presets.py (e.g. preset1, preset2, preset3, ...)
        interactive (optional): bool (default: False)
            start a pype, modify loaded preset before saving it to phenopype directories
        overwrite (optional): bool (default: False)
            overwrite option, if a given pype config-file already exist
        """

        ## kwargs
        pype_name = kwargs.get("name","1")
        preset = kwargs.get("preset","preset1")
        flag_interactive = kwargs.get("interactive", None)
        flag_overwrite = kwargs.get("overwrite", False)
        steps_include = kwargs.get("include",[]) 

        ## modify
        if flag_interactive:
            container = load_directory(self.dirlist[0])
            config, template_path = _generic_pype_config(container, preset = preset)
            template_path = os.path.join(self.root_dir, "pype_template.yaml")
            _save_yaml(config, template_path)
            p = pype(self.rawlist[0], steps=steps_include, config=template_path)
            config = p.config
        else:
            config = _load_yaml(eval("presets." + preset))

        ## go through project directories
        for directory in self.dirlist:
            attr = _load_yaml(os.path.join(directory, "attributes.yaml"))
            pype_preset = {"image": attr["image"],
                           "pype":
                               {"name": pype_name,
                                "preset": preset,
                                "date_created": datetime.today().strftime('%Y%m%d_%H%M%S'),
                                "date_last_used": None}}
            pype_preset.update(config)

            ## save config
            preset_path = os.path.join(directory, "pype_" + pype_name + ".yaml")
            dirname = attr["project"]["dirname"]
            if os.path.isfile(preset_path) and flag_overwrite==False:
                print("pype_" + pype_name + ".yaml already exists in " + dirname +  " (overwrite=False)")
                continue
            elif os.path.isfile(preset_path) and flag_overwrite==True:
                print("pype_" + pype_name + ".yaml created for " + dirname + " (overwritten)")
                _save_yaml(pype_preset, preset_path)
            else:
                print("pype_" + pype_name + ".yaml created for " + dirname)
                _save_yaml(pype_preset, preset_path)



class pype:
    def __init__(self, obj_input, **kwargs):
        """
        Pype method. 
        
        Parameters
        ----------

        name (optional): str (default: "v1")
            name of config-file
        preset (optional): str (default: "preset1")
            chose from given presets in phenopype/settings/pype_presets.py (e.g. preset1, preset2, preset3, ...)
        interactive (optional): bool (default: False)
            start a pype, modify loaded preset before saving it to phenopype directories
        overwrite (optional): bool (default: False)
            overwrite option, if a given pype config-file already exist
        """
        ## kwargs
        flag_return = kwargs.get("return_results",False)
        flag_show = kwargs.get("show",True)
        steps_exclude = kwargs.get("exclude",[]) 
        steps_include = kwargs.get("include",[]) 
        config = kwargs.get("config", None)
        print_settings = kwargs.get("print_settings",False)        
        default_fields = ["DateTimeOriginal","Model","LensModel","ExposureTime", "ISOSpeedRatings","FNumber"]
        exif_fields = kwargs.get("fields", default_fields)
        if not exif_fields.__class__.__name__ == "list":
            exif_fields = [exif_fields]

        ## load image as cointainer from array, file, or directory
        if obj_input.__class__.__name__ == "ndarray":
            self.container = load_image(obj_input, container=True, meta=False)
        elif obj_input.__class__.__name__ == "str":
            if os.path.isfile(obj_input):
                self.container = load_image(obj_input, container=True, meta=True, fields=exif_fields)
            elif os.path.isdir(obj_input):
                self.container = load_directory(obj_input, meta=True, fields=exif_fields)
        else:
            sys.exit("Wrong input - cannot run pype.")

        ## load config
        if config:
            self.config, self.config_file = _load_pype_config(self.container, config=config)
        else: 
            self.config, self.config_file = _load_pype_config(self.container)
        _save_yaml(self.config, self.config_file)

        ## open config file with system viewer
        if flag_show:
            if platform.system() == 'Darwin':       # macOS
                subprocess.call(('open', self.config_file))
            elif platform.system() == 'Windows':    # Windows
                os.startfile(self.config_file)
            else:                                   # linux variants
                subprocess.call(('xdg-open', self.config_file))

        ## initialize
        self.FM = _yaml_file_monitor(self.config_file, print_settings=print_settings)
        update = {}
        iv = None
        
        # =============================================================================
        # pype
        # =============================================================================
        
        while True:
            ## check visulization given
            flag_vis = False
            
            ## get config file and assemble pype
            self.config = copy.deepcopy(self.FM.content)
            if not self.config:
                continue
            
            ## check steps
            if not steps_include:
                steps_pre = []
                for item in self.config:
                    steps_pre.append(item)
                steps = [e for e in steps_pre if e not in steps_exclude]
            elif steps_include:
                steps = steps_include

            ## apply pype
            for step in steps:
                if step in ["image", "meta", "pype"]:
                    continue
                for item in self.config[step]:
                    try:
                        ## construct method name and arguments
                        if isinstance(item, str):
                            method_name = item
                            method_args = {}
                        else:
                            method_name = list(item)[0]
                            method_args = dict(list(dict(item).values())[0])
                            
                        ## run method
                        method_loaded = eval(step + "." + method_name)
                        method_loaded(self.container, **method_args)

                        ## check if visualization argument is given in config
                        if method_name == "show_image":
                            flag_vis = True
                    except Exception as ex:
                        location = step + "." + method_name + ": " + str(ex.__class__.__name__)
                        print(location + " - " + str(ex))

            ## show image and hold
            if flag_show:
                try:
                    if not flag_vis:
                        self.container = visualization.show_image(self.container)
                    iv = _image_viewer(self.container.canvas, prev_attributes=update)

                    ## pass on settings for next call
                    update = iv.__dict__

                except Exception as ex:
                    print("visualisation: " + str(ex.__class__.__name__) + " - " + str(ex))

                ## close
                if iv:
                    if iv.done:
                        self.FM.stop()
                        break
            else:
                self.FM.stop()
                break
            
            ## reset container
            self.container.reset(components=["contours"])
            print("\n\n---------------new pype iteration---------------\n\n")

        if flag_return:
            return self.container



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
