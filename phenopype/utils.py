#%% modules
import ast, cv2, copy, os, sys, warnings
import json
import numpy as np
import glob
import pandas as pd
import pkgutil
import string
from dataclasses import make_dataclass

from pathlib import Path
from datetime import datetime
import io
from contextlib import redirect_stdout
import ruamel.yaml

import phenopype.core.preprocessing as preprocessing
import phenopype.core.segmentation as segmentation
import phenopype.core.measurement as measurement
import phenopype.core.visualization as visualization
import phenopype.core.export as export

from phenopype.settings import (
    default_filetypes,
    flag_verbose, 
    confirm_options, 
    _annotation_types,
    )
from phenopype import utils_lowlevel 
from collections import defaultdict

#%% classes

class Container(object):
    """
    A phenopype container is a Python class where loaded images, dataframes, 
    detected contours, intermediate output, etc. are stored so that they are 
    available for inspection or storage at the end of the analysis. The 
    advantage of using containers is that they don’t litter the global environment 
    and namespace, while still containing all intermediate steps (e.g. binary 
    masks or contour DataFrames). Containers can be used manually to analyse images, 
    but typically they are created dynamically within the pype-routine. 
    
    Parameters
    ----------
    image_path : ndarray
        path to an image to build the container from. Image directory will be used
        as the working-directory to store all Pype-output in
    version : str, optional
        suffix to append to filename of results files. The default is "v1".

    """

    def __init__(self, image, dir_path, **kwargs):

        ## images
        self.image = image
        self.image_copy = copy.deepcopy(self.image)
        self.canvas = None

        ## attributes
        self.file_prefix = kwargs.get("file_prefix")
        self.file_suffix = kwargs.get("file_suffix")
        self.dir_path = dir_path
        self.image_name = kwargs.get("image_name")
        ## annotations - primed from empty dict
        self.annotations = copy.deepcopy(_annotation_types)
                
        
    def load(self, contours=False,  **kwargs):
        """
        Autoload function for container: loads results files with given file_suffix
        into the container. Can be used manually, but is typically used within the
        pype routine.
        
        Parameters
        ----------
        file_suffix : str, optional
            suffix to include when looking for files to load

        """

        loaded = []        

        ## load annotations
        
        annotations_file_name = self._construct_file_name("annotations",".json")

        if annotations_file_name in os.listdir(self.dir_path):
            annotations_loaded = export.load_annotation(os.path.join(self.dir_path, annotations_file_name))
            if annotations_loaded:
                self.annotations.update(annotations_loaded)
                                            
            self.annotations["data"] = {}
            
            annotation_types_loaded = {}
            for annotation_type in self.annotations.keys():
                id_list = []
                for annotation_id in self.annotations[annotation_type].keys():
                    id_list.append(annotation_id)
                if len(id_list) > 0:
                    annotation_types_loaded[annotation_type] = utils_lowlevel._NoIndent(id_list)
                
            loaded.append("annotations loaded:\n{}".format(json.dumps(annotation_types_loaded, indent=0, cls=utils_lowlevel._NoIndentEncoder)))

            
        ## load attributes
        attr_local_path = os.path.join(self.dir_path, "attributes.yaml")
        if os.path.isfile(attr_local_path):

            self.image_attributes = utils_lowlevel._load_yaml(attr_local_path)
            
            if "reference" in self.image_attributes:
                
                ## manually measured px-mm-ratio
                if "reference_manually_measured_px_mm_ratio" in self.image_attributes["reference"]:
                    self.reference_manually_measured_px_mm_ratio = self.image_attributes["reference"]["reference_manually_measured_px_mm_ratio"]
                    loaded.append("manually measured local reference information loaded")
                    
                ## project level template px-mm-ratio
                if "project_level" in self.image_attributes["reference"]:
                    
                    ## load local (image specific) and global (project level) attributes 
                    attr_proj_path =  os.path.abspath(os.path.join(attr_local_path ,r"../../../","attributes.yaml"))
                    attr_proj = utils_lowlevel._load_yaml(attr_proj_path)
                                        
                    ## find active project level references
                    n_active = 0
                    for key, value in self.image_attributes["reference"]["project_level"].items():
                        if self.image_attributes["reference"]["project_level"][key]["active"] == True:
                            active_ref = key
                            n_active += 1
                    if n_active > 1:
                        print("WARNING: multiple active reference detected - fix with running add_reference again.")                            
                    self.reference_active = active_ref
                    self.reference_template_px_mm_ratio = attr_proj["reference"][active_ref]["template_px_mm_ratio"]
                    loaded.append("project level reference information loaded for " + active_ref)
                
                    ## load previously detect px-mm-ratio
                    if "reference_detected_px_mm_ratio" in self.image_attributes["reference"]["project_level"][active_ref]:
                        self.reference_detected_px_mm_ratio = self.image_attributes["reference"]["project_level"][active_ref]["reference_detected_px_mm_ratio"]
                        loaded.append("detected local reference information loaded for " + active_ref)
                        
                    ## load tempate image from project level attributes
                    if "template_image" in attr_proj["reference"][active_ref]:
                        self.reference_template_image = cv2.imread(str(Path(attr_local_path).parents[2] / attr_proj["reference"][active_ref]["template_image"]))
                        loaded.append("reference template image loaded from root directory")       
                
        ## feedback
        if len(loaded) > 0:
            print("\nAUTOLOAD\n- " + "\n- ".join(loaded))
        else:
            print("\nAUTOLOAD\n - nothing to autoload")

            
    def reset(self):
        """
        Resets modified images, canvas and df_image_data to original state. Can be used manually, but is typically used within the
        pype routine.

        """
        ## images
        self.image = copy.deepcopy(self.image_copy)
        # self.image_bin = None
        self.image_gray = None
        self.canvas = None
        # self.annotations = None

            
    def run(self, fun, fun_kwargs={}, annotation_kwargs={}, annotation_counter={}):
        
        ## annotation kwargs
        annotation = None
        annotation_id = annotation_kwargs.get("id") 
        annotation_type = annotation_kwargs.get("type")
        edit = annotation_kwargs.get("edit", False)
        
        ## annotation_counter
        fun_kwargs["annotation_counter"] = annotation_counter
        
        ## function kwargs
        kwargs = fun_kwargs
        
        
        ## attributes
        if hasattr(self, "image_attributes"):
            image_name=self.image_attributes["image_original"]["filename"] 
        elif self.image_name:
            image_name = self.image_name
                
        ## edit handling
        if not all(
                [annotation_id.__class__.__name__ == "NoneType",
                 annotation_type.__class__.__name__ == "NoneType"]):
            if annotation_id in self.annotations[annotation_type]:
                print_msg = "- loaded existing annotation of type \"{}\" with ID \"{}\"".format(annotation_type, annotation_id)
                if edit == True:
                    kwargs.update({"annotation_previous":self.annotations[annotation_type][annotation_id]})
                    print(print_msg + ": editing (edit=True)")
                elif edit == False:
                    kwargs.update({"annotation_previous":self.annotations[annotation_type][annotation_id]})
                    print(print_msg + ": skipping (edit=False)")
                    if annotation_type in ["drawing"]:
                        kwargs.update({"passive":True})                                             
                        self.image, annotation = segmentation.edit_contour(self.canvas, annotation=self.annotations, **kwargs)
                    return
                elif edit == "overwrite":
                    print(print_msg + ": overwriting (edit=overwrite)")
                    pass
                
        ## preprocessing
        if fun == "blur":
            self.image = preprocessing.blur(self.image, **kwargs)
        if fun == "create_mask":
            annotation = preprocessing.create_mask(self.image, **kwargs)
        if fun == "detect_shape":
            annotation = preprocessing.detect_shape(self.image, **kwargs)
        if fun == "comment":
            annotation = preprocessing.comment(self.image, **kwargs)
        if fun == "detect_reference":
            if not any(hasattr(self, reference) for reference in [
                    "reference_detected_px_mm_ratio"
                    ]):
                if all(hasattr(self, template) for template in [
                        "reference_template_px_mm_ratio", 
                        "reference_template_image"
                        ]):
                    annotation = preprocessing.detect_reference(
                        self.image, 
                        self.reference_template_image,
                        self.reference_template_px_mm_ratio,
                        **kwargs)
                    if annotation.__class__.__name__ == "tuple":
                        ref_mask_id = string.ascii_lowercase[len(self.annotations["mask"].keys())]
                        self.annotations["mask"][ref_mask_id] = annotation[1]
                        annotation = annotation[0]
                    self.reference_as_detected = self.annotations[annotation_type][annotation_id]["data"]["px_mm_ratio"]
                else:
                    print("- missing project level reference information, cannot detect")
            else:
                print("- reference already detected (current px-to-mm-ratio: {}).".format(self.reference_detected_px_mm_ratio)) 
        if fun == "decompose_image":
            self.image = preprocessing.decompose_image(self.image, **kwargs)
            
        ## segmentation
        if fun == "threshold":
            if "mask" in self.annotations and len(self.annotations["mask"]) > 0 :
                kwargs.update({"mask":self.annotations["mask"]})
            self.image = segmentation.threshold(self.image, **kwargs)
            # self.image_bin = copy.deepcopy(self.image)
        # if fun == "watershed":
        #     self.image = segmentation.watershed(self.image_copy, self.image_bin, **kwargs)
        if fun == "morphology":
            self.image = segmentation.morphology(self.image, **kwargs)
        if fun == "detect_contour":
            annotation = segmentation.detect_contour(self.image, **kwargs)
        if fun == "edit_contour":
            if self.canvas.__class__.__name__ == "NoneType":
                visualization.select_canvas(self)
            image, annotation = segmentation.edit_contour(self.canvas, annotation=self.annotations, **kwargs)
            self.image = image
            
        ## measurement
        if fun == "set_landmark":
            annotation = measurement.set_landmark(self.canvas, **kwargs)
        if fun == "set_polyline":
            annotation = measurement.set_polyline(self.canvas, **kwargs)
        if fun == "skeletonize":
            annotation = measurement.skeletonize(self.image, annotation=self.annotations, **kwargs)
        if fun == "shape_features":
            annotation = measurement.shape_features(annotation=self.annotations, **kwargs)
        if fun == "texture_features":
            annotation = measurement.texture_features(self.image_copy, annotation=self.annotations, **kwargs)

        ## visualization
        if fun == "select_canvas":
            visualization.select_canvas(self, **kwargs)
        if fun == "draw_contour":
            self.canvas = visualization.draw_contour(self.canvas, annotation=self.annotations, **kwargs)
        if fun == "draw_landmark":
            self.canvas = visualization.draw_landmark(self.canvas, annotation=self.annotations, **kwargs)
        if fun == "draw_mask":
            self.canvas = visualization.draw_mask(self.canvas, annotation=self.annotations, **kwargs)
        if fun == "draw_polyline":
            self.canvas = visualization.draw_polyline(self.canvas, annotation=self.annotations, **kwargs)
            
        ## export
        if fun == "save_annotation":
            export.save_annotation(self.annotations, 
                                   file_name=self._construct_file_name("annotations","json"), 
                                   dir_path=self.dir_path, 
                                   **kwargs)        
        if fun == "save_canvas":
            export.save_canvas(self.canvas, 
                               file_name=self._construct_file_name("canvas",kwargs.get("ext",".jpg")), 
                               dir_path=self.dir_path, 
                               **kwargs)
        if fun == "export_csv":                
            export.export_csv(self.annotations, 
                              dir_path=self.dir_path, 
                              save_prefix = self.file_prefix,
                              save_suffix = self.file_suffix,
                              image_name = image_name,
                              **kwargs)           
            
        ## save annotation to dict
        if annotation:
            self.annotations[annotation_type][annotation_id] = annotation

            
    def save(self, dir_path=None, export_list=[], overwrite=False, **kwargs):
        """
        Autosave function for container. 
        
        Parameters
        ----------
        dir_path: str, optional
            provide a custom directory where files should be save - overwrites 
            dir_path provided from container, if applicable
        export_list: list, optional
            used in pype rountine to check against already performed saving operations.
            running container.save() with an empty export_list will assumed that nothing
            has been saved so far, and will try 
        overwrite : bool, optional
            gloabl overwrite flag in case file exists
            
        """

        ## kwargs
        flag_overwrite = overwrite
        flag_autosave = False

        ## check dir_path
        print("\nAUTOSAVE")
        
        if hasattr(self, "canvas") and not "save_canvas" in export_list:
            print("- save_canvas")
            export.save_canvas(self.canvas, 
                               dir_path=self.dir_path, 
                               **kwargs)
            flag_autosave = True

        if hasattr(self, "annotations") and not "save_annotation" in export_list:
            print("- save_annotation")
            export.save_annotation(self.annotations, 
                                   file_name=self._construct_file_name("annotations","json"), 
                                   dir_path=self.dir_path, 
                                   **kwargs)
            flag_autosave = True

        if "reference" in self.annotations and not len(self.annotations["reference"]) == 0 and not "reference" in export_list:
            print("- save reference")
            export.save_reference(self.annotations, 
                                  dir_path=dir_path, 
                                  overwrite=flag_overwrite, 
                                  active_ref=self.reference_active)
            flag_autosave = True

        if not flag_autosave:
            print("- nothing to autosave")
            
    def _construct_file_name(self, stem, ext):
        
        if not ext.startswith("."):
            ext = "." + ext
         
        if self.file_prefix:
            prefix = self.file_prefix + "_"
        else:
            prefix = ""
        if self.file_suffix:
            suffix = "_" + self.file_suffix
        else:
            suffix = ""
            
        return prefix + stem + suffix + ext

#%% functions


def load_image(
    path,
    mode="default",
    **kwargs
):
    """
    Create ndarray from image path or return or resize exising array.

    Parameters
    ----------
    path: str
        path to an image stored on the harddrive
    mode: {"default", "colour","gray"} str, optional
        image conversion on loading:
            - default: load image as is
            - colour: convert image to 3-channel (BGR)
            - gray: convert image to single channel (grayscale)
    kwargs: 
        developer options

    Returns
    -------
    container: container
        A phenopype container is a Python class where loaded images, 
        dataframes, detected contours, intermediate output, etc. are stored 
        so that they are available for inspection or storage at the end of 
        the analysis. 
    image: ndarray
        original image (resized, if selected)

    """
    
    ## set flags
    flags = make_dataclass(cls_name="flags", 
                           fields=[("mode", str, mode)])   
     
    ## load image
    if path.__class__.__name__ == "str":
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext.replace(".", "") in default_filetypes:
                if flags.mode == "default":
                    image = cv2.imread(path)
                elif flags.mode == "colour":
                    image = cv2.imread(path, cv2.IMREAD_COLOR)
                elif flags.mode == "gray":
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)     
            else:
                raise OSError("Invalid file extension \"{}\" - could not load image:\n".format(ext))
                return
        else:
            raise FileNotFoundError("Invalid image path - could not load image.")
            # return
    else:
        raise FileNotFoundError("Invalid input format - could not load image.")
        return            
            
    return image
   


def load_template(
        template_path,
        tag = "v1",
        overwrite=False,
        keep_comments=True,
        image_path=None,
        dir_path=None,
        ret_path=False,
        ):
    
    flags = make_dataclass(cls_name="flags", 
                           fields=[("overwrite", bool, overwrite)])   
   
    ## create config from template
    if template_path.__class__.__name__ == "str": 
        if os.path.isfile(template_path):
            template_loaded = utils_lowlevel._load_yaml(template_path)
        else:
            raise FileNotFoundError("Could not find template_path")
    else:
        raise TypeError("Wrong input format for template_path")
    
    ## construct config-name
    if dir_path.__class__.__name__ == "NoneType" and image_path.__class__.__name__ == "NoneType":
        raise AttributeError("Need to specify image_path or dir_path")
    elif dir_path.__class__.__name__ == "str" and image_path.__class__.__name__ == "NoneType":
        if os.path.isdir(dir_path):
            prepend = ""
        else:
            raise FileNotFoundError("Could not find dir_path")
    elif dir_path.__class__.__name__ == "NoneType":    
        dir_path = os.path.dirname(image_path)
        image_name_root = os.path.splitext(os.path.basename(image_path))[0]
        prepend = image_name_root + "_"
        
    if tag.__class__.__name__ == "str":
        suffix = "_" + tag
    else:
        suffix = ""
        
    config_name = prepend + "pype_config" + suffix + ".yaml"
    config_path = os.path.join(dir_path, config_name) 

    ## strip template name
    if "template" in template_loaded:
        template_loaded.pop("template")

    
    config_info = {
        "config_info": {
            "config_name":config_name,
            "date_created":datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "date_last_modified":None,
            "template_name":os.path.basename(template_path),
            "template_path":template_path,
            }
        }
    
    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=4, sequence=4, offset=4)

    if keep_comments == True:
    
        with io.StringIO() as buf, redirect_stdout(buf):
            yaml.dump(config_info, sys.stdout)
            output = buf.getvalue()
            output = yaml.load(output)
            
        for key in reversed(output):  
            template_loaded.insert(0, key, output[key])

    else:
        template_loaded = {**config_info, **template_loaded}
        utils_lowlevel._yaml_recursive_delete_comments(template_loaded)
               
    if utils_lowlevel._save_prompt("template",config_path, flags.overwrite):
        with open(config_path, "wb") as yaml_file:
            yaml.dump(template_loaded, yaml_file)
            
    if ret_path:
        return config_path

            

def save_image(
    image,
    file_name,
    dir_path,
    resize=1,
    ext="jpg",
    overwrite=False,
    verbose=True,
    **kwargs
):
    """Save an image (array) to jpg.
    
    Parameters
    ----------
    image: array
        image to save
    name: str
        name for saved image
    save_dir: str, optional
        directory to save image
    append: str, optional
        append image name with string to prevent overwriting
    extension: str, optional
        file extension to save image as
    overwrite: boo, optional
        overwrite images if name exists
    resize: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of
        original size).
    kwargs: 
        developer options
    """

    ## kwargs
    flag_overwrite = overwrite

    # set dir and names
    # if "." in name:
    #     warnings.warn("need name and extension specified separately")
    #     return

    if "." not in ext:
        ext = "." + ext
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    ## resize
    if resize < 1:
        image = cv2.resize(
            image, (0, 0), fx=1 * resize, fy=1 * resize, interpolation=cv2.INTER_AREA
        )

    ## construct save path
    
    if file_name.endswith(ext):
        new_name = file_name
    else:
        new_name = file_name + ext
    path = os.path.join(dir_path, new_name)

    ## save
    while True:
        print_msg = None
        if os.path.isfile(path) and flag_overwrite == False:
            print_msg = "- image not saved - file already exists (overwrite=False)."
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print_msg =  "- image saved under " + path + " (overwritten)."
            pass
        elif not os.path.isfile(path):
            print_msg =  "- image saved under " + path + "."
            pass
        
        if verbose:
            print(print_msg)
        
        cv2.imwrite(path, image)
        break



def show_image(
    image,
    window_max_dim=1200,
    position_reset=True,
    position_offset=25,
    window_aspect="normal",
    check=True,
    **kwargs
):
    """
    Show one or multiple images by providing path string or array or list of 
    either.
    
    Parameters
    ----------
    image: array, list of arrays
        the image or list of images to be displayed. can be array-type, 
        or list or arrays
    window_max_dim: int, optional
        maximum dimension on either acis
    window_aspect: {"fixed", "free"} str, optional
        type of opencv window ("free" is resizeable)
    position_reset: bool, optional
        flag whether image positions should be reset when reopening list of 
        images
    position_offset: int, optional
        if image is list, the distance in pixels betweeen the positions of 
        each newly opened window (only works in conjunction with 
        "position_reset")
    check: bool, optional
        user input required when more than 10 images are opened at the same 
        time
    """
    ## kwargs
    flag_check = check

    ## load image
    if image.__class__.__name__ == "ndarray":
        pass
    elif image.__class__.__name__ == "list":
        pass
    else:
        print("wrong input format.")
        return

    ## open images list or single images
    while True:
        if isinstance(image, list):
            if len(image) > 10 and flag_check == True:
                warning_string = (
                    "WARNING: trying to open "
                    + str(len(image))
                    + " images - proceed (y/n)?"
                )
                check = input(warning_string)
                if check in ["y", "Y", "yes", "Yes"]:
                    print("Proceed - Opening images ...")
                    pass
                else:
                    print("Aborting")
                    break
            idx = 0
            for i in image:
                idx += 1
                if i.__class__.__name__ == "ndarray":
                    print("phenopype" + " - " + str(idx))
                    utils_lowlevel._ImageViewer(
                        i,
                        mode="",
                        window_aspect=window_aspect,
                        window_name="phenopype" + " - " + str(idx),
                        window_control="external",
                        window_max_dim=window_max_dim,
                    )
                    if position_reset == True:
                        cv2.moveWindow(
                            "phenopype" + " - " + str(idx),
                            idx + idx * position_offset,
                            idx + idx * position_offset,
                        )
                else:
                    print("skipped showing list item of type " + i.__class__.__name__)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        else:
            utils_lowlevel._ImageViewer(
                image=image,
                mode="",
                window_aspect=window_aspect,
                window_name="phenopype",
                window_control="internal",
                window_max_dim=window_max_dim,
                # window_max_dim=window_max_dim,
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        