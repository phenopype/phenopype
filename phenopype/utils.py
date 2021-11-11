#%% modules
import ast, cv2, copy, os, sys, warnings
import numpy as np
import glob
import pandas as pd
import pkgutil
import string
from dataclasses import make_dataclass

from pathlib import Path

import phenopype.core.preprocessing as preprocessing
import phenopype.core.segmentation as segmentation
import phenopype.core.measurement as measurement
import phenopype.core.visualization as visualization
import phenopype.core.export as export

from phenopype.settings import default_filetypes, flag_verbose, confirm_options, _annotation_types
from phenopype.utils_lowlevel import _ImageViewer, _convert_tup_list_arr,  _load_image_data, _load_yaml, _show_yaml
    
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
    image : ndarray
        single or multi-channel iamge as an array (can be created using load_image 
        or load_pp_directory).
    df_image_data: DataFrame
        a dataframe that contains meta-data of the provided image to be passed on
        to all results-DataFrames
    save_suffix : str, optional
        suffix to append to filename of results files

    """

    def __init__(self, image, dirpath=None, save_suffix=None):

        ## images
        self.image = image
        self.image_copy = copy.deepcopy(self.image)
        # self.image_bin = None
        self.image_gray = None
        self.canvas = None

        ## attributes
        self.dirpath = dirpath
        self.save_suffix = save_suffix
        
        ## annotations
        self.annotations = copy.deepcopy(_annotation_types)
                
        
    def load(self, contours=False,  **kwargs):
        """
        Autoload function for container: loads results files with given save_suffix
        into the container. Can be used manually, but is typically used within the
        pype routine.
        
        Parameters
        ----------
        save_suffix : str, optional
            suffix to include when looking for files to load

        """

        loaded = []        

        ## load annotations
        annotations_filename = "annotations" + "_" + self.save_suffix + ".json"
        if annotations_filename in os.listdir(self.dirpath):
            self.annotations.update(export.load_annotation(os.path.join(self.dirpath, annotations_filename)))
            loaded.append("annotations loaded")
                    
        if contours == False:
            self.annotations["contour"] = {}
            
        ## load attributes
        attr_local_path = os.path.join(self.dirpath, "attributes.yaml")
        if os.path.isfile(attr_local_path):

            attr_local = _load_yaml(attr_local_path)
            
            if "reference" in attr_local:
                
                ## manually measured px-mm-ratio
                if "reference_manually_measured_px_mm_ratio" in attr_local["reference"]:
                    self.reference_manually_measured_px_mm_ratio = attr_local["reference"]["reference_manually_measured_px_mm_ratio"]
                    loaded.append("manually measured local reference information loaded")
                    
                ## project level template px-mm-ratio
                if "project_level" in attr_local["reference"]:
                    
                    ## load local (image specific) and global (project level) attributes 
                    attr_proj_path =  os.path.abspath(os.path.join(attr_local_path ,r"../../../","attributes.yaml"))
                    attr_proj = _load_yaml(attr_proj_path)
                                        
                    ## find active project level references
                    n_active = 0
                    for key, value in attr_local["reference"]["project_level"].items():
                        if attr_local["reference"]["project_level"][key]["active"] == True:
                            active_ref = key
                            n_active += 1
                    if n_active > 1:
                        print("WARNING: multiple active reference detected - fix with running add_reference again.")                            
                    self.reference_active = active_ref
                    self.reference_template_px_mm_ratio = attr_proj["reference"][active_ref]["template_px_mm_ratio"]
                    loaded.append("project level reference information loaded for " + active_ref)
                
                    ## load previously detect px-mm-ratio
                    if "reference_detected_px_mm_ratio" in attr_local["reference"]["project_level"][active_ref]:
                        self.reference_detected_px_mm_ratio = attr_local["reference"]["project_level"][active_ref]["reference_detected_px_mm_ratio"]
                        loaded.append("detected local reference information loaded for " + active_ref)
                        
                    ## load tempate image from project level attributes
                    if "template_image" in attr_proj["reference"][active_ref]:
                        self.reference_template_image = cv2.imread(str(Path(attr_local_path).parents[2] / attr_proj["reference"][active_ref]["template_image"]))
                        loaded.append("reference template image loaded from root directory")       
                
        ## feedback
        if len(loaded) > 0:
            print("=== AUTOLOAD ===\n- " + "\n- ".join(loaded))
        else:
            print("AUTOLOAD off - nothing loaded.")

            
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
                
        if not all(
                [annotation_id.__class__.__name__ == "NoneType",
                 annotation_type.__class__.__name__ == "NoneType"]):
            if annotation_id in self.annotations[annotation_type]:
                if edit == True:
                    kwargs.update({"annotation_previous":self.annotations[annotation_type][annotation_id]})
                    print("- editing annotation of type \"{}\" with ID \"{}\" already present (edit=True)".format(annotation_type, annotation_id))
                elif edit == False:
                    kwargs.update({"annotation_previous":self.annotations[annotation_type][annotation_id]})
                    print("- annotation of type \"{}\" with ID \"{}\" already present (overwrite=False)".format(annotation_type, annotation_id))
                    if annotation_type == "drawing":
                        kwargs.update({"passive":True})                                             
                        self.image, annotation = segmentation.edit_contour(self.canvas, annotation=self.annotations, **kwargs)
                    return
                elif edit == "overwrite":
                    print("- overwriting annotation of type \"{}\" with ID \"{}\" already present (edit=overwrite)".format(annotation_type, annotation_id))
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
            image, annotation = segmentation.edit_contour(self.canvas, annotation=self.annotations, **kwargs)
            self.image = image

        ## visualization
        if fun == "select_canvas":
            visualization.select_canvas(self, **kwargs)
        if fun == "draw_contour":
            self.canvas = visualization.draw_contour(self.canvas, annotation=self.annotations, **kwargs)
        if fun == "draw_landmark":
            self.canvas = visualization.draw_landmark(self.canvas, annotation=self.annotations, **kwargs)
        if fun == "draw_mask":
            self.canvas = visualization.draw_mask(self.canvas, annotation=self.annotations, **kwargs)
            
        ## export
        if fun == "save_annotation":
            filename = kwargs.get("filename", "annotations") + "_" + self.save_suffix + ".json"
            export.save_annotation(self.annotations, dirpath=self.dirpath, filename=filename, **kwargs)
        if fun == "save_canvas":
            export.save_canvas(self.canvas, dirpath=self.dirpath, save_suffix=self.save_suffix, **kwargs)
            
            
        ## save annotation to dict
        if annotation:
            self.annotations[annotation_type][annotation_id] = annotation

            
    def save(self, dirpath=None, export_list=[], overwrite=False, **kwargs):
        """
        Autosave function for container. 
        Parameters
        ----------
        dirpath: str, optional
            provide a custom directory where files should be save - overwrites 
            dirpath provided from container, if applicable
        export_list: list, optional
            used in pype rountine to check against already performed saving operations.
            running container.save() with an empty export_list will assumed that nothing
            has been saved so far, and will try 
        overwrite : bool, optional
            gloabl overwrite flag in case file exists
        """

        ## kwargs
        flag_overwrite = overwrite

        ## check dirpath
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not self.dirpath.__class__.__name__ == "NoneType"
        ):
            print("=== AUTOSAVE ===")
            dirpath = self.dirpath
        if dirpath.__class__.__name__ == "NoneType":
            print('No save directory ("dirpath") specified - cannot save files.')
            return
        if not os.path.isdir(dirpath):
            print("Directory does not exist - cannot save files.")


        # ## canvas
        # if (
        #     not self.canvas.__class__.__name__ == "NoneType"
        #     and not "save_canvas" in export_list
        # ):
        #     print("save_canvas")
        #     export.save_canvas(self, dirpath=dirpath)

        if hasattr(self, "annotations") and not "save_annotation" in export_list:
            print("- save_annotation")
            filename = kwargs.get("filename", "annotations") + "_" + self.save_suffix + ".json"            
            export.save_annotation(self.annotations, dirpath=self.dirpath, filename=filename, **kwargs)

        if "reference" in self.annotations and not len(self.annotations["reference"]) == 0 and not "reference" in export_list:
            print("- save reference")
            export.save_reference(self.annotations, dirpath=dirpath, overwrite=flag_overwrite, active_ref=self.reference_active)
            

#%% functions


def load_image(
    path,
    mode="default",
    load_container=False,
    dirpath=None,
    save_suffix=None,
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
    load_container: bool, optional
        should the loaded image (and DataFrame) be returned as a phenopype 
        container
    dirpath: str, optional
        path to an existing directory where all output should be stored. default 
        is the current working directory ("cwd") of the python session.
    save_suffix : str, optional
        suffix to append to filename of results files, if container is created
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
                           fields=[("mode", str, mode), 
                                   ("load_container", bool, load_container)])   
     
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
                print("Invalid file extension \"{}\" - could not load image:\n".format(ext) \
                        + os.path.basename(path))
                return
        else:
            print("Invalid image path - could not load image.")
            return
    elif path.__class__.__name__ == "ndarray":
        image = path
    else:
        print("Invalid input format - could not load image.")
        return

    ## check dirpath
    if flags.load_container == True:
        if dirpath == "cwd":
            dirpath = os.getcwd()
            if flag_verbose:
                print(
                    "Setting directory to save phenopype-container output to current working directory:\n" \
                    + os.path.abspath(dirpath)
                )
        elif dirpath.__class__.__name__ == "str":
            if not os.path.isdir(dirpath):
                user_input = input(
                    "Provided directory to save phenopype-container output {} does not exist - create?.".format(
                        os.path.abspath(dirpath)
                    )
                )
                if user_input in confirm_options:
                    os.makedirs(dirpath)
                else:
                    print("Directory not created - aborting")
                    return
            else:
                if flag_verbose:
                    print("Directory to save phenopype-container output set at - " + os.path.abspath(dirpath))
        elif dirpath.__class__.__name__ == "NoneType":
            if path.__class__.__name__ == "str":
                if os.path.isfile(path):
                    dirpath = os.path.dirname(os.path.abspath(path))
                    if flag_verbose:
                        print("Directory to save phenopype-container output set to parent folder of image:\n{}".format(dirpath))
            else: 
                print(
                    "No directory provided to save phenopype-container output" +
                    " - provide dirpath or use dirpath==\"cwd\" to set save" +
                    " paths to current working directory - aborting."
                      )
                return
            
            
    ## create container
    if flags.load_container:
        cont = copy.deepcopy(Container(image, dirpath=dirpath, save_suffix=save_suffix))
        return cont
    else:
        return image


def load_pp_directory(
    dirpath, 
    load_container=True, 
    save_suffix=None, 
    **kwargs
):
    """
    Parameters
    ----------
    dirpath: str or ndarray
        path to a phenopype project directory containing raw image, attributes 
        file, masks files, results df, etc.
    cont: bool, optional
        should the loaded image (and DataFrame) be returned as a phenopype 
        container
    save_suffix : str, optional
        suffix to append to filename of results files
    kwargs: 
        developer options
        
    Returns
    -------
    container
        A phenopype container is a Python class where loaded images, 
        dataframes, detected contours, intermediate output, etc. are stored 
        so that they are available for inspection or storage at the end of 
        the analysis. 

    """
    
    ## set flags
    flags = make_dataclass(cls_name="flags", 
                           fields=[ ("load_container", bool, load_container)])   
    ## check if directory
    if not os.path.isdir(dirpath):
        print("Not a valid phenoype directory - cannot load files.")
        return
    
    ## check if attributes file and load otherwise
    if not os.path.isfile(os.path.join(dirpath, "attributes.yaml")):
        print("Attributes file missing - cannot load files.")
        return
    else:
        attributes = _load_yaml(os.path.join(dirpath, "attributes.yaml"))
    
    ## check if requires info is contained in attributes and load image
    if not "image_phenopype" in attributes or not "image_original" in attributes:
        print("Attributes doesn't contain required meta-data - cannot load files.")
        return 

    ## load image
    if attributes["image_phenopype"]["mode"] == "link":
        image_path =  attributes["image_original"]["filepath"]
    else:
        image_path =  os.path.join(dirpath,attributes["image_phenopype"]["filename"])
        
    ## return
    return load_image(image_path, load_container=flags.load_container, dirpath=dirpath, save_suffix=save_suffix)
    


def save_image(
    image,
    name,
    dirpath=os.getcwd(),
    resize=1,
    append="",
    extension="jpg",
    overwrite=False,
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
    if append == "":
        append = ""
    else:
        append = "_" + append
    if "." not in extension:
        extension = "." + extension
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    ## resize
    if resize < 1:
        image = cv2.resize(
            image, (0, 0), fx=1 * resize, fy=1 * resize, interpolation=cv2.INTER_AREA
        )

    ## construct save path
    new_name = name + append + extension
    path = os.path.join(dirpath, new_name)

    ## save
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("Image not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("Image saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("Image saved under " + path + ".")
            pass
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
    elif image.__class__.__name__ == "container":
        if not image.canvas.__class__.__name__ == "NoneType":
            image = copy.deepcopy(image.canvas)
        else:
            image = copy.deepcopy(image.image)
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
                    _ImageViewer(
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
            _ImageViewer(
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
        