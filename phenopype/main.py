#%%
import copy
import cv2
import datetime
import os
import numpy as np
import pickle
import pandas as pd
import platform
import pprint
import ruamel.yaml
import shutil
import subprocess
import sys 

from shutil import copyfile
from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.utils import yaml_file_monitor, load_yaml, save_yaml
from phenopype.utils_lowlevel import _image_viewer, _del_rw, _make_pype_template
from phenopype.settings import pype_presets, colours
from phenopype import preprocessing, segmentation, extraction, visualization

#%% settings

pd.options.display.max_rows = 10 # how many rows of pd-dataframe to show
pretty = pprint.PrettyPrinter(width=30) # pretty print short strings
ruamel.yaml.Representer.add_representer(ordereddict, ruamel.yaml.Representer.represent_dict) # suppress !!omap node info


#%% methods

class project: 
    """
    Initialize a phenopype project with a name.
    
    Parameters
    ----------

    root_dir: str (default: "CurrentWorkingDir + "CurrentDate_phenopype")
        root directory of the project
    name: str (default: "CurrentDate_phenopype")
        name of your project
    """
    
    def __init__(self, **kwargs):

        flag_overwrite = kwargs.get("overwrite", False)
        flag_query = kwargs.get("query", True)
        
        if "root_dir" in kwargs:
            root_dir = kwargs.get("root_dir")
            if os.path.basename(root_dir) == "":
                name = kwargs.get("name", datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + "_project")
                root_dir = os.path.join(root_dir, name)
            else:
                name = os.path.basename(root_dir)
        elif "name" in kwargs:
            name = kwargs.get("name")
            root_dir = os.path.join(os.getcwd(), name)
        else:
            name = datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + "_phenopype_project"
            root_dir = os.path.join(os.getcwd(), name)
            
        
        print("\n")
        print("--------------------------------------------")
        print("phenopype will create a new project named \"" + name + "\". \
              The full path of the project's root directory will be:\n")
        print(root_dir + ".phenopype")
        
        while True:
            if flag_query == False:
                create = "y"
            else:
                create = input("Proceed? (y/n)\n")
            
            if create=="y" or create == "yes":
                if os.path.isdir(root_dir + ".phenopype"):
                    if flag_overwrite == True:
                        shutil.rmtree(root_dir + ".phenopype", ignore_errors=True, onerror=_del_rw) 
                        print("\n\"" + root_dir + ".phenopype\" created (overwritten)")
                        pass
                    else:
                        overwrite = input("Warning - project already exists - overwrite? (y/n)")
                        if overwrite == "y" or overwrite == "yes":
                            shutil.rmtree(root_dir + ".phenopype", ignore_errors=True, onerror=_del_rw) 
                            print("\n\"" + root_dir + ".phenopype\" created (overwritten)")
                            pass
                        else:
                            print("\n\"" + name + "\" not created!")
                            print("--------------------------------------------")    
                            break
                else:
                    pass
            
            self.name = name
            self.filenames = []  
            self.filelist = []
            self.filebinder = {}
            
            self.root_dir = os.path.abspath(root_dir) + ".phenopype"   
            os.mkdir(self.root_dir)
            self.data_dir = os.path.join(self.root_dir, "data")
            os.mkdir(self.data_dir)
                                    
            config = {"date_created": datetime.datetime.today().strftime('%Y%m%d_%H%M%S'),
                      "date_changed": datetime.datetime.today().strftime('%Y%m%d_%H%M%S')}
            with open(os.path.join(self.root_dir, "attributes.yaml"), 'w') as config_file:
                yaml = ruamel.yaml.YAML()
                yaml.dump(config, config_file) 
                
            print("\n\"" + name + "\" created at \"" + root_dir + ".phenopype\"")
            print("--------------------------------------------")
            break


                    
    def add_files(self, image_dir, **kwargs):
        """Add files to your project from a directory, can look recursively. 
        Optional: specify a search string for filenames or file extensions. 
    
        Parameters
        ----------
    
        image_dir: str 
            path to directory with images                             
        search_mode: str (default: "dir")
            "dir" searches current directory for valid files; "recursive" walks through all subdirectories
        filetypes: list 
            single or multiple string patterns to target files with certain endings
        include: list 
            single or multiple string patterns to target certain files to include - can be used together with exclude
        exclude: list 
            single or multiple string patterns to target certain files to include - can be used together with include
        """
        
        ## kwargs                     
        flag_overwrite = kwargs.get("overwrite",False)
        search_mode = kwargs.get("search_mode","dir")                 
        file_endings = kwargs.get("filetypes", [])
        exclude_args = kwargs.get("exclude", [])
        include_args = kwargs.get("include", [])
        unique_by = kwargs.get("unique_by", "filepaths")
        resize_raw = kwargs.get("resize_raw", 1)
        
        ## dummy filepaths for refinement
        filepaths1, filepaths2, filepaths3, filepaths4 = [],[],[],[]
        filepaths_original, filepaths_not_added = [], []
        ## find files 
        if search_mode == "recursive":
            for root, dirs, files in os.walk(image_dir):
                for file in os.listdir(root):
                    filepath = os.path.join(root,file)
                    if os.path.isfile(filepath):
                        filepaths1.append(filepath)
        elif search_mode == "dir":
            for file in os.listdir(image_dir):
                filepath = os.path.join(image_dir,file)
                if os.path.isfile(filepath):   
                    filepaths1.append(filepath)
                    
        ## file endings
        if len(file_endings)>0:
            for filepath in filepaths1:
                if filepath.endswith(tuple(file_endings)):
                    filepaths2.append(filepath)
        elif len(file_endings)==0:
            filepaths2 = filepaths1
            
        ## include
        if len(include_args)>0:
            for filepath in filepaths2:   
                if any(inc in os.path.basename(filepath) for inc in include_args):
                    filepaths3.append(filepath)
        else:
            filepaths3 = filepaths2
            
        ## exclude
        if len(exclude_args)>0:
            for filepath in filepaths3:   
                if not any(exc in os.path.basename(filepath) for exc in exclude_args):
                    filepaths4.append(filepath)
        else:
            filepaths4 = filepaths3
        
        ## save to object
        filepaths = filepaths4
        filenames = []
        for filepath in filepaths:
            filenames.append(os.path.basename(filepath))       
        
        ## allow unique filenames filepath or by filename only
        if unique_by=="filepaths":
            for filename, filepath in zip(filenames, filepaths):
                if not filepath in filepaths_original:
                    filepaths_original.append(filepath)
                else:
                    filepaths_not_added.append(filepath)
        elif unique_by=="filenames":
            for filename, filepath in zip(filenames, filepaths):
                if not filename in filenames:
                    filepaths_original.append(filepath)
                else:
                    filepaths_not_added.append(filepath)
            
        ## loop through files
        for filepath_original in filepaths_original:
            
            ## flatten folder structure
            subfolder_prefix = os.path.dirname(os.path.relpath(filepath_original,image_dir)).replace("\\","__")
            filename = subfolder_prefix + "____" + os.path.basename(filepath_original)
            filepath = os.path.join(self.root_dir,"data",filename)
            
            ## make folder 
            if os.path.isdir(filepath) and flag_overwrite==False:
                print(filename + " already exists (overwrite=False)")
                continue
            if os.path.isdir(filepath) and flag_overwrite==True:
                shutil.rmtree(filepath, ignore_errors=True, onerror=_del_rw)
                print(filename + "phenopype-project folder " + filename + " created (overwritten)")
                pass
            else:
                print("phenopype-project folder " + filename + " created")
                pass
            
            ## make image-specific directories
            os.mkdir(filepath)
            
            ## copy and rename raw-file
            filepath_raw = os.path.join(filepath,"raw" + ".jpg") # os.path.splitext(filename)[1]
            
            copyfile(filepath_original, filepath_raw)

            ## specify attributes
            attributes = {
                "filename": filename,
                "filepath": filepath,
                "filetype": os.path.splitext(filename)[1],
                "filepath_original": filepath_original,
                "filepath_raw": filepath_raw,
                "filepath_attributes": os.path.join(filepath, "attributes.yaml")
                }
                
            ## add to project object
            self.filenames.append(filename)
            self.filelist.append(attributes)
            self.filebinder[filename] = attributes

        # ## update config file
        # with open(self.config_filepath, 'r') as config_file:
        #     config = yaml.load(config_file)
        # config["date_changed"] = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        # config["filepaths_original"] = self.filepaths_original
        # with open(self.config_filepath, 'w') as config_file:
        #     config = yaml.dump(config, config_file) 
                                    
    def add_pype_preset(self, **kwargs):

        ## kwargs
        flag_overwrite = kwargs.get("overwrite", False)
        flag_interactive = kwargs.get("test_and_modify", None)
        pype_name = kwargs.get("name","pipeline1")
        preset_name = kwargs.get("preset","config1")

        pype_config = _make_pype_template(name=pype_name, preset=preset_name)

        ## modify preset 
        if flag_interactive:
            if isinstance(flag_interactive, str):
                if os.path.isfile(flag_interactive):
                    pype_path = os.path.join(os.path.dirname(flag_interactive), pype_name + ".yaml")
                    save_yaml(pype_config, pype_path)
                    p = pype(flag_interactive)
                elif os.path.isdir(flag_interactive):
                    pype_path = os.path.join(flag_interactive, pype_name + ".yaml")
                    save_yaml(pype_config, pype_path)
                    p = pype(os.path.join(flag_interactive, "raw.jpg"))
            if isinstance(flag_interactive, dict):
                pype_path = os.path.join(flag_interactive["filepath"], pype_name + ".yaml")
                save_yaml(pype_config, pype_path)
                p = pype(flag_interactive["filepath_raw"])
            elif isinstance(flag_interactive, bool): 
                if flag_interactive==True:
                    pype_path = os.path.join(self.filelist[0]["filepath"], pype_name + ".yaml")
                    save_yaml(pype_config, pype_path)
                    p = pype(self.filelist[0]["filepath_raw"])
            p.run(steps=["segmentation"], pype_config=pype_path)
            pype_config = p.pype_config

        ## save pype-config file
        for file in self.filelist:
            filename = file["filename"]
            filepath = file["filepath"]
            pype_path = os.path.join(filepath, pype_name + ".yaml")
            print(pype_path)
            if os.path.isfile(pype_path) and flag_overwrite==False:
                print(pype_name + ".yaml already exists (overwrite=False)")
                continue
            elif os.path.isfile(pype_path) and flag_overwrite==True:
                print(pype_name + ".yaml created for " + filename + " (overwritten)")
                pass
            else:
                print(pype_name + ".yaml created for " + filename)
                pass
            save_yaml(pype_config, pype_path)
                
def project_save(project_file):
    output_str = os.path.join(project_file.root_dir, 'project.data')
    with open(output_str, 'wb') as output:
        pickle.dump(project_file, output, pickle.HIGHEST_PROTOCOL)
         
def project_load(path):
    with open(path, 'rb') as output:
        return pickle.load(output)

class pype:
    def __init__(self, image, **kwargs):

        ## load image 
        if isinstance(image, str):
            if os.path.isfile(image):
                name = os.path.basename(os.path.dirname(image))
                filepath = image
                dirname = os.path.dirname(image)
                image = cv2.imread(image)
            elif os.path.isdir(image):
                name = os.path.basename(image)
                filepath = os.path.join(image, "raw.jpg")
                dirname = image
                image = cv2.imread(filepath)     
        elif isinstance(image, dict):
            name = image["filename"]
            filepath = image["filepath_raw"]
            dirname = image["filepath"]
            image = cv2.imread(filepath)
        elif isinstance(image, np.ndarray):
            name = kwargs.get("name","img1.jpg")
            filepath = os.getcwd()
            dirname = os.getcwd()
            image = image
        
        self.name = name
        self.filepath = filepath
        self.dirname = dirname
        self.image = image
        
        ## create pype container from image or directory path, or array 
        self.PC = pype_container(self.image)
            
    def run(self, **kwargs):
        
        ## kwargs
        print_settings = kwargs.get("print_settings",False)
        flag_return = kwargs.get("return_results",False)   
        steps_exclude = kwargs.get("exclude",[]) 
        steps_include = kwargs.get("include",[]) 

        ## fetch pype configuration from file or preset
        pype_config = kwargs.get("pype_config", None)
    
        if isinstance(pype_config,  str):
            pype_config_locations = [os.path.join(self.dirname, pype_config + ".yaml"),
                       os.path.join(self.dirname, pype_config)]
            for loc in pype_config_locations:
                if os.path.isfile(loc):
                    self.pype_config = load_yaml(loc)
                    self.pype_config_file = loc
        else:
            self.pype_config = pype_config
            self.pype_config_file = self.dirname
            
        if not pype_config:
            print("not")
            loc = os.path.join(self.dirname, "pipeline1.yaml")
            save_yaml(_make_pype_template(), loc)
            self.pype_config = load_yaml(loc)
            self.pype_config_file = loc

        ## open config file with system viewer
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', self.pype_config_file))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(self.pype_config_file)
        else:                                   # linux variants
            subprocess.call(('xdg-open', self.pype_config_file))

        self.FM = yaml_file_monitor(self.pype_config_file, print_settings=print_settings)
        update = {}
        iv = None
        
        while True:
            
            ## reset pype container
            flag_vis = False
            self.PC.reset(components=["contour_list"])
            
            ## get config file and assemble pype
            self.pype_config = self.FM.content
            if not steps_include:
                steps_pre = []
                for item in self.pype_config:
                    steps_pre.append(item)
                steps = [e for e in steps_pre if e not in steps_exclude]
            elif steps_include:
                steps = steps_include

            ## apply pype
            for step in steps:
                if step == "pype_header":
                    continue
                for item in self.pype_config[step]:
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
                        method_loaded(self.PC, **method_args)

                        ## check if visualization argument is given in config
                        if method_name == "show_image":
                            flag_vis = True
                            
                    except Exception as ex:
                        print(step + "." + method_name + ": " + str(ex.__class__.__name__) + " - " + str(ex))

            ## show image and hold
            try:
                if not flag_vis:
                    self.PC = visualization.show_image(self.PC)
                iv = _image_viewer(self.PC.canvas, prev_attributes=update)
    
                ## pass on settings for next call
                update = iv.__dict__

            except Exception as ex:
                print("visualisation: " + str(ex.__class__.__name__) + " - " + str(ex))

            ## close
            if iv:
                if iv.done:
                    self.FM.stop()
                    break

        if flag_return:
            return self.PC



class pype_container(object):
    def __init__(self, image):
        """ This is a pype-data object where the image and other data 
         is stored that can be passed on between pype-steps
        
        Parameters
        ----------
    
        image: str or array
            image can be provided as path to file or diretory (phenpype dir), as phenopype-binder object (dict), 
            or as numpy array
        """
        
        ## images
        self.image = image
        self.image_mod = copy.deepcopy(self.image)
        self.image_bin = None
        self.canvas = None
        
        self.contour_list = []
        self.contour_hierarchy = []   
        self.mask_binder = {}
        
    def reset(self, components=[]):
        
        self.image_mod = copy.deepcopy(self.image)
        self.image_bin = None
        self.canvas = None

        if "contour" in components or "contours" in components or "contour_list" in components:
            self.contour_list = []
            self.contour_hierarchy = []
        if "mask" in components or "masks" in components:
            self.mask_binder = {}
            
            
        # def save(self):     
        #     with open(self.path + '.data', 'wb') as output:
        #         pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        # def load(self):
        #     with open(self.content + '.data', 'rb') as output:
        #         pickle.load(output)
                
                
        #     self.mask = {}
        # def show(self):
        #     pretty.pprint(self.__dict__)
        # def load_raw(self):
        #     img = cv2.imread(os.path.join(self.filepath_raw))
        #     return img
        # def show_raw(self):
        #     img = cv2.imread(os.path.join(self.filepath_raw))
        #     show_img(img)
            
        # def create_mask(self, **kwargs):
        #     name = kwargs.get("name", "mask1")
        #     self.mask[name] = mask.create_mask(self.filepath_raw)
        #     cv2.imwrite(name + ".jpg", self.mask[name].mask_bin)
        
            
# class image_container(object):    
#     def __init__(self, filename, filepath): 
#         self.filename = filename    
#         self.filetype = os.path.splitext(filename)[1]
#         self.filepath = filepath    
#         self.filepath_raw = os.path.join(filepath,"raw" + self.filetype)
#         self.filepath_config = os.path.join(filepath,"config.yaml")
        