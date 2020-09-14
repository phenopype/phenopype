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
from phenopype.settings import (
    default_filetypes,
    default_pype_config,
    default_meta_data_fields,
)
from phenopype.core import preprocessing, segmentation, measurement, export, visualization
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

pd.options.display.max_rows = 10  # how many rows of pd-dataframe to show
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
            self.data_dir = os.path.join(self.root_dir, "data")
            os.makedirs(self.data_dir)

            # ##  set working directory
            # if not os.path.abspath(root_dir) == os.getcwd():
            #     os.chdir(root_dir)
            #     print("Current working directory changed to " + os.path.abspath(root_dir))
            # else:
            #     print("Already in " + os.path.abspath(root_dir))

            ## generate empty lists
            for lst in [
                "dirnames",
                "dirpaths_rel",
                "dirpaths",
                "filenames",
                "filepaths_rel",
                "filepaths",
            ]:
                setattr(self, lst, [])

            ## global project attributes
            project_data = {
                "date_created": datetime.today().strftime("%Y%m%d_%H%M%S"),
                "date_changed": datetime.today().strftime("%Y%m%d_%H%M%S"),
                "root_dir": self.root_dir,
                "data_dir": self.data_dir,
            }

            _save_yaml(project_data, os.path.join(self.root_dir, "attributes.yaml"))

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
        exclude=[],
        raw_mode="copy",
        search_mode="dir",
        unique_mode="path",
        overwrite=False,
        resize=1,
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
        exclude: list or str, optional
            single or multiple string patterns to target certain files to exclude - 
            can overrule "include"
        raw_mode: {"copy", "link"} str, optional
            how should the raw files be passed on to the phenopype directory tree: 
            "copy" will make a copy of the original file, "link" will only send the 
            link to the original raw file to attributes, but not copy the actual 
            file (useful for big files)
        search_mode: {"dir", "recursive"}, str, optional
            "dir" searches current directory for valid files; "recursive" walks 
            through all subdirectories
        unique_mode: {"filepath", "filename"}, str, optional:
            how to deal with image duplicates - "filepath" is useful if identically 
            named files exist in different subfolders (folder structure will be 
            collapsed and goes into the filename), whereas filename will ignore 
            all similar named files after their first occurrence.
        kwargs: 
            developer options
        """

        # kwargs
        flag_raw_mode = raw_mode
        flag_overwrite = overwrite
        flag_resize = resize

        ## path conversion
        image_dir = image_dir.replace(os.sep, "/")
        image_dir = os.path.abspath(image_dir)

        ## collect filepaths
        filepaths, duplicates = _file_walker(
            directory=image_dir,
            search_mode=search_mode,
            unique_mode=unique_mode,
            filetypes=filetypes,
            exclude=exclude,
            include=include,
        )

        ## feedback
        print("--------------------------------------------")
        print("phenopype will search for files at\n")
        print(image_dir)
        print("\nusing the following settings:\n")
        print(
            "filetypes: "
            + str(filetypes)
            + ", include: "
            + str(include)
            + ", exclude: "
            + str(exclude)
            + ", raw_mode: "
            + str(raw_mode)
            + ", search_mode: "
            + str(search_mode)
            + ", unique_mode: "
            + str(unique_mode)
            + "\n"
        )

        ## loop through files
        for filepath in filepaths:

            ## generate phenopype dir-tree
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

            ## load image
            image = load_image(filepath, resize=flag_resize)
            fileending = os.path.splitext(os.path.basename(filepath))[1]
            
            ## copy or link raw files
            if flag_raw_mode == "copy":
                raw_path = os.path.join(
                    self.data_dir,
                    dirname,
                    "raw" + fileending,
                )
                if resize < 1:
                    cv2.imwrite(raw_path, image)
                else:
                    copyfile(filepath, raw_path)

            elif flag_raw_mode == "link":
                if resize < 1:
                    warnings.warn("cannot resize image in link mode")
                raw_path = filepath

            ## path reformatting
            dir_relpath = os.path.relpath(dirpath, self.root_dir)
            dir_relpath = dir_relpath.replace(os.sep, "/")
            raw_relpath = os.path.join(dir_relpath, "raw" + fileending)

            ## collect attribute-data and save
            image_data = load_image_data(filepath, flag_resize)
            meta_data = load_meta_data(filepath)
            project_data = {
                "dirname": dirname,
                "dirpath": dir_relpath,
                "raw_mode": flag_raw_mode,
                "raw_path": raw_relpath,
            }

            if meta_data:
                attributes = {
                    "image": image_data,
                    "meta": meta_data,
                    "project": project_data,
                }
            else:
                attributes = {"image": image_data, "project": project_data}

            ## write attributes file
            _save_yaml(
                attributes, os.path.join(self.root_dir, dir_relpath, "attributes.yaml")
            )

            ## add to project object
            if not dirname in self.dirnames:
                
                ## directories
                self.dirnames.append(dirname)
                self.dirpaths_rel.append(dir_relpath)
                self.dirpaths.append(os.path.join(self.root_dir, dir_relpath))
                
                ## files
                self.filenames.append(image_data["filename"])
                self.filepaths_rel.append(raw_relpath)
                self.filepaths.append(os.path.join(self.root_dir, raw_relpath))

        print("\nFound {} files".format(len(filepaths)))
        print("--------------------------------------------")

    def add_config(
        self,
        name,
        config_preset=None,
        interactive=False,
        overwrite=False,
        idx=0,
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

        ## kwargs
        flag_interactive = interactive
        flag_overwrite = overwrite

        ## legacy
        preset = kwargs.get("preset")
        if (
            config_preset.__class__.__name__ == "NoneType"
            and not preset.__class__.__name__ == "NoneType"
        ):
            config_preset = preset

        ## load config
        if not config_preset.__class__.__name__ == "NoneType" and hasattr(
            presets, config_preset
        ):
            config = _create_generic_pype_config(preset=config_preset, config_name=name)
        elif not config_preset.__class__.__name__ == "NoneType" and os.path.isfile(
            config_preset
        ):
            config = {
                "pype": {
                    "name": name,
                    "preset": config_preset,
                    "date_created": datetime.today().strftime("%Y%m%d_%H%M%S"),
                }
            }
            config.update(_load_yaml(config_preset))
            print(config)
        elif not config_preset.__class__.__name__ == "NoneType" and not hasattr(
            presets, config_preset
        ):
            print("Provided preset NOT found - terminating")
            return
        elif config_preset.__class__.__name__ == "NoneType":
            print("No preset provided - defaulting to preset " + default_pype_config)
            config = _load_yaml(eval("presets." + default_pype_config))

        ## modify
        if flag_interactive:
            image_location = os.path.join(
                self.root_dir,
                "pype_template_image" + os.path.splitext(self.filenames[idx])[1],
            )
            copyfile(self.filepaths[idx], image_location)
            config_location = os.path.join(
                self.root_dir, "pype_config_template-" + name + ".yaml"
            )
            _save_yaml(config, config_location)
            p = pype(
                image_location,
                name="template-" + name,
                config_location=config_location,
                presetting=True,
            )
            config = p.config

        ## go through project directories
        for directory in self.dirpaths:
            attr = _load_yaml(os.path.join(self.root_dir, directory, "attributes.yaml"))
            pype_preset = {"image": attr["image"]}
            pype_preset.update(config)

            ## save config
            preset_path = os.path.join(
                self.root_dir, directory, "pype_config_" + name + ".yaml"
            )
            dirname = attr["project"]["dirname"]
            if os.path.isfile(preset_path) and flag_overwrite == False:
                print(
                    "pype_"
                    + name
                    + ".yaml already exists in "
                    + dirname
                    + " (overwrite=False)"
                )
                continue
            elif os.path.isfile(preset_path) and flag_overwrite == True:
                print("pype_" + name + ".yaml created for " + dirname + " (overwritten)")
                _save_yaml(pype_preset, preset_path)
            else:
                print("pype_" + name + ".yaml created for " + dirname)
                _save_yaml(pype_preset, preset_path)

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
        step, 
        function,
        **kwargs
    ):
        """
        [new/experimental] Add or edit functions in all configuration files of a project.

        Parameters
        ----------

        name: str
            name of config-file. this gets appended to all files and serves as and
            identifier of a specific analysis pipeline
        step: str
            name of the step the function is in 
        function: str
            name of the function
        """

        ## kwargs
        flag_checked = False
        
        ## go through project directories
        for directory in self.dirpaths:
            dirname = os.path.basename(directory)

            ## save config
            preset_path = os.path.join(
                self.root_dir, directory, "pype_config_" + name + ".yaml"
            )
            
            if os.path.isfile(preset_path):
                config = _load_yaml(preset_path)
                
            ordered_steps = ["preprocessing",
                          "segmentation",
                          "measurement",
                          "visualization",
                          "export"
                          ]

            if not step in config.keys():
                new_config = ordereddict([("image", ordereddict(config["image"]))])
                new_config.update(ordereddict([("pype", ordereddict(config["pype"]))]))
                for ordered_step in ordered_steps:
                    if ordered_step in config:
                        new_config.update(ordereddict([(ordered_step, config[ordered_step])]))
                    elif not ordered_step in config and ordered_step == step:
                        new_config.update(ordereddict([(ordered_step, [function] )]))
            else:
                new_config = copy.deepcopy(config)
                if not function in new_config[step]:
                    new_config[step].append(function)
                    
            
            if flag_checked == False:
                _show_yaml(new_config)
                check = input("This is what the new config may look like (can differ beteeen files) - proceed?")
            
            if check in ["True", "true", "y", "yes"]:
                flag_checked = True
                _save_yaml(new_config, preset_path)
                print("New config saved for " + dirname)
            else:
                print("User check failed - aborting.")
                return 
                
                
                
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
    def load(path):
        """
        Load phenoype project.data file to Python namespace.
    
        Parameters
        ----------
        path: str
            path to project.data, or the directory project.data is contained in
            (root_dir)

        Returns
        -------
        project: project
            phenopype project object
        """

        if "project.data" in path:
            pass
        else:
            path = os.path.join(path, "project.data")
        with open(path, "rb") as output:
            proj = pickle.load(output)

        ## path conversion
        proj.root_dir = os.path.split(path)[0]
        proj.root_dir = proj.root_dir.replace(os.sep, "/")
        proj.root_dir = os.path.abspath(proj.root_dir)
        proj.data_dir = os.path.join(os.path.abspath(proj.root_dir), "data")

        # ##  set working directory
        # if not proj.root_dir == os.getcwd():
        #     os.chdir(proj.root_dir)

        ## legacy
        if not hasattr(proj, "dirpath_rel"):
            proj.dirpaths_rel, proj.filepaths_rel = [], []
        for dirname, filename in zip(proj.dirnames, proj.filenames):
            proj.dirpaths_rel.append(os.path.join("data", dirname))
            proj.filepaths_rel.append(os.path.join("data", dirname, filename))

        ## set correct paths
        proj.dirpaths, proj.filepaths = [], []
        for dirpath_rel, filepath_rel in zip(proj.dirpaths_rel, proj.filepaths_rel):
            proj.dirpaths.append(os.path.join(proj.root_dir, dirpath_rel))
            proj.filepaths.append(os.path.join(proj.root_dir, filepath_rel))

        ## feedback
        print("--------------------------------------------")
        print("Project loaded from \n" + proj.root_dir)
        print("--------------------------------------------")

        return proj


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
                print(self.config_location)
                os.startfile(self.config_location)
            else:  # linux variants
                subprocess.call(("xdg-open", self.config_location))

        ## initialize
        self.FM = _yaml_file_monitor(self.config_location, delay)
        update, terminate, iv = {}, False, None

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

            ## apply pype
            for step in list(self.config.keys()):
                if step in ["image", "meta", "pype"]:
                    continue
                if not self.config[step]:
                    continue
                if step == "export" and flag_presetting == True:
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
                            if (
                                not "select_canvas" in show_list
                                and self.container.canvas.__class__.__name__ == "NoneType"
                            ):
                                visualization.select_canvas(self.container)
                                print("- autoselect canvas")

                        ## run method
                        print(method_name)
                        method_loaded = eval(step + "." + method_name)
                        restart = method_loaded(self.container, **method_args)

                        ## control
                        if restart:
                            print("RESTART")
                            break

                    except Exception as ex:
                        location = (
                            step + "." + method_name + ": " + str(ex.__class__.__name__)
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
                    self.container.save(export_list=export_list)

            ## visualize output
            if flag_feedback:
                try:
                    if self.container.canvas.__class__.__name__ == "NoneType":
                        visualization.select_canvas(self.container)
                        print("- autoselect canvas")
                    if flag_test_mode:
                        update = test_params
                    iv = _image_viewer(self.container.canvas, previous=update)
                    update, terminate = iv.__dict__, iv.done
                except Exception as ex:
                    print(
                        "visualisation: " + str(ex.__class__.__name__) + " - " + str(ex)
                    )

            ## terminate
            if terminate or not flag_feedback:
                self.FM.stop()
                print("\n\nTERMINATE")
                break
