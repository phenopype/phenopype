#%% modules

clean_namespace = dir()

import copy
import cv2
import io
import os
import sys
import json
import webbrowser

from pathlib import Path
from datetime import datetime
from dataclasses import make_dataclass
from contextlib import redirect_stdout
from pkg_resources import resource_filename
import ruamel.yaml

from phenopype import _config
from phenopype import assets
from phenopype import core
from phenopype import plugins
from phenopype import settings
from phenopype import utils_lowlevel

classes = ["Container"]
functions = ['load_image', "load_template", "print_colours", "save_image", "show_image"]

def __dir__():
    return clean_namespace + classes + functions

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

        ## set reference image
        self.image_copy = image

        ## assign copies
        self.image = copy.deepcopy(self.image_copy)
        self.canvas = copy.deepcopy(self.image_copy)

        ## attributes (needs more order/cleaning)
        self.tag = kwargs.get("tag")
        self.file_prefix = kwargs.get("file_prefix")
        self.file_suffix = kwargs.get("file_suffix")
        self.dir_path = dir_path
        self.image_name = kwargs.get("image_name")

        ## annotations
        self.annotations = {}

    def load(self, contours=False, **kwargs):
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
        annotations_file_name = self._construct_file_name("annotations", ".json")
        
        if annotations_file_name in os.listdir(self.dir_path):
            annotations_loaded = core.export.load_annotation(
                os.path.join(self.dir_path, annotations_file_name)
            )
            
            if contours == False:
                if settings._contour_type in annotations_loaded:
                    annotations_loaded.pop(settings._contour_type)
            
            if annotations_loaded:
                self.annotations.update(annotations_loaded)

            self.annotations["data"] = {}

            annotation_types_loaded = {}
            for annotation_type in self.annotations.keys():
                id_list = []
                for annotation_id in self.annotations[annotation_type].keys():
                    id_list.append(annotation_id)
                if len(id_list) > 0:
                    annotation_types_loaded[annotation_type] = utils_lowlevel._NoIndent(
                        id_list
                    )

            loaded.append(
                "annotations loaded:\n{}".format(
                    json.dumps(
                        annotation_types_loaded,
                        indent=0,
                        cls=utils_lowlevel._NoIndentEncoder,
                    )
                )
            )

        ## global attributes
        attr_proj_path = os.path.abspath(
            os.path.join(self.dir_path, r"../../", "attributes.yaml")
        )
        if os.path.isfile(attr_proj_path):
            self.attr_proj = utils_lowlevel._load_yaml(attr_proj_path)

        ## load attributes
        attr_local_path = os.path.join(self.dir_path, "attributes.yaml")
        if os.path.isfile(attr_local_path):

            self.image_attributes = utils_lowlevel._load_yaml(attr_local_path)

            if "reference_global" in self.image_attributes:

                ## load local (image specific) and global (project level) attributes
                attr_proj_path = os.path.abspath(
                    os.path.join(attr_local_path, r"../../../", "attributes.yaml")
                )
                attr_proj = utils_lowlevel._load_yaml(attr_proj_path)

                ## find active project level references
                n_active = 0
                for key, value in self.image_attributes["reference_global"].items():
                    if self.image_attributes["reference_global"][key]["active"] == True:
                        active_ref = key
                        n_active += 1
                if n_active > 1:
                    print(
                        "WARNING: multiple active reference detected - fix with running add_reference again."
                    )

                self.reference_active = active_ref

                ## load tempate image from project level attributes
                if "template_file_name" in attr_proj["reference"][active_ref]:
                    self.reference_template_image = cv2.imread(
                        str(
                            Path(attr_local_path).parents[2]
                            / "reference"
                            / attr_proj["reference"][active_ref]["template_file_name"]
                        )
                    )
                    self.reference_template_px_ratio = attr_proj["reference"][
                        active_ref
                    ]["template_px_ratio"]
                    self.reference_unit = attr_proj["reference"][active_ref]["unit"]

                    loaded.append("reference template image loaded from root directory")
                    
            if "models" in self.attr_proj:
                               
                if len(self.attr_proj["models"]) > 0:
                    for model_info in self.attr_proj["models"].items():
                        if not model_info[0] in _config.models:
                            _config.models[model_info[0]] = model_info[1]
                        else:
                            _config.models[model_info[0]].update(model_info[1])    
                    loaded.append("loaded info for {} models {} ".format(len(_config.models.keys()),(*list(_config.models.keys()),)))
                        
                if "active" in self.attr_proj:
                    if "model" in self.attr_proj["active"]:
                        model_id = self.attr_proj["active"]["model"]
                        model_path = self.attr_proj["models"][model_id]["model_phenopype_path"]
                        _config.active_model_path = model_path    
                        loaded.append('set model "{}" as default model (change with "activate=True")'.format(model_id))


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

        ## re-assign copies
        self.image = copy.deepcopy(self.image_copy)
        self.canvas = copy.deepcopy(self.image_copy)

    def run(
            self, 
            fun,
            fun_kwargs={}, 
            annotation_kwargs={}, 
            annotation_counter={},
            ):

        ## annotation kwargs
        annotations = copy.deepcopy(self.annotations)
        annotation_type = annotation_kwargs.get("type")
        annotation_id = annotation_kwargs.get("id")

        flag_edit = annotation_kwargs.get("edit", False)
        annotations_updated = None

        ## function kwargs
        kwargs_function = copy.deepcopy(fun_kwargs)
        kwargs_function["annotations"] = annotations
        kwargs_function["annotation_type"] = fun_kwargs.get("annotation_type",annotation_type)
        kwargs_function["annotation_id"] = fun_kwargs.get("annotation_id",annotation_id)
        kwargs_function["annotation_counter"] = annotation_counter

        ## use pype tag
        kwargs_function["tag"] = self.tag

        ## verbosity
        if settings.flag_verbose:
            kwargs_function["verbose"] = True

        ## enable zoom-config memory
        kwargs_function["pype_mode"] = True

        ## attributes
        if hasattr(self, "image_attributes"):
            image_name = self.image_attributes["image_original"]["filename"]
        elif self.image_name:
            image_name = self.image_name

        ## edit handling
        if not all([
                annotation_id.__class__.__name__ == "NoneType",
                annotation_type.__class__.__name__ == "NoneType",
            ]):
            if annotation_type in annotations:
                if annotation_id in annotations[annotation_type]:
                    print_msg = '- loaded existing annotation of type "{}" with ID "{}"'.format(
                        annotation_type, annotation_id
                    )
                    if flag_edit == True:
                        print(print_msg + ": editing (edit=True)")
                    elif flag_edit == False:
                        print(print_msg + ": skipping (edit=False)")
                        if annotation_type in ["drawing"]:
                            kwargs_function["feedback"] = False
                            annotations_updated, self.image = core.segmentation.edit_contour(
                                self.canvas, ret_image=True, **kwargs_function
                            )
                        return
                    elif flag_edit == "overwrite":
                        print(print_msg + ": overwriting (edit=overwrite)")
                        annotations[annotation_type][annotation_id] = {}
                        pass

        ## preprocessing
        if fun == "blur":
            self.image = core.preprocessing.blur(self.image, **kwargs_function)
        if fun == "create_mask":
            annotations_updated = core.preprocessing.create_mask(self.image, **kwargs_function)
        if fun == "create_reference":
            annotations_updated = core.preprocessing.create_reference(self.image, **kwargs_function)
        if fun == "detect_mask":
            annotations_updated = core.preprocessing.detect_mask(self.image, **kwargs_function)
            # print(annotations_updated)
        if fun == "write_comment":
            annotations_updated = core.preprocessing.write_comment(self.image, **kwargs_function)
        if fun == "detect_reference":
            if all(
                hasattr(self, attr)
                for attr in [
                    "reference_template_px_ratio",
                    "reference_template_image",
                    "reference_unit",
                ]
            ):
                annotations_updated = core.preprocessing.detect_reference(
                    self.image,
                    self.reference_template_image,
                    self.reference_template_px_ratio,
                    self.reference_unit,
                    **kwargs_function,
                )
            else:
                print("- missing project level reference information, cannot detect")
        if fun == "decompose_image":
            self.image = core.preprocessing.decompose_image(self.image, **kwargs_function)

        ## plugins.segmentation
        if fun == "detect_object":
            # if len(self.image.shape) == 2:
            #     self.image = copy.deepcopy(self.image_copy)
            self.image = plugins.segmentation.detect_object(self.image_copy, _config.active_model_path, **kwargs_function)

        ## core.segmentation
        if fun == "contour_to_mask":
            annotations_updated = core.segmentation.contour_to_mask(**kwargs_function)
        if fun == "threshold":
            self.image = core.segmentation.threshold(self.image, **kwargs_function)
        if fun == "watershed":
            self.image = core.segmentation.watershed(self.image, **kwargs_function)
        if fun == "morphology":
            self.image = core.segmentation.morphology(self.image, **kwargs_function)
        if fun == "detect_contour":
            annotations_updated = core.segmentation.detect_contour(self.image, **kwargs_function)
        if fun == "edit_contour":
            annotations_updated, self.image = core.segmentation.edit_contour(
                self.canvas, ret_image=True, **kwargs_function
            )

        ## core.measurement
        if fun == "set_landmark":
            annotations_updated = core.measurement.set_landmark(image=self.canvas, **kwargs_function)
        if fun == "set_polyline":
            annotations_updated = core.measurement.set_polyline(self.canvas, **kwargs_function)
        if fun == "detect_skeleton":
            annotations_updated = core.measurement.detect_skeleton(**kwargs_function)
        if fun == "compute_shape_features":
            annotations_updated = core.measurement.compute_shape_features(**kwargs_function)
        if fun == "compute_texture_features":
            annotations_updated = core.measurement.compute_texture_features(
                self.image_copy, **kwargs_function
            )

        ## plugins.measurement
        if fun == "detect_landmark":
            annotations_updated = plugins.measurement.detect_landmark(
                image = self.image,
                model_path = self.active_model_path,
                **kwargs_function)

        ## visualization
        if fun == "select_canvas":
            core.visualization.select_canvas(self, **kwargs_function)
        if fun == "draw_contour":
            self.canvas = core.visualization.draw_contour(self.canvas, **kwargs_function)
        if fun == "draw_landmark":
            self.canvas = core.visualization.draw_landmark(self.canvas, **kwargs_function)
        if fun == "draw_mask":
            self.canvas = core.visualization.draw_mask(self.canvas, **kwargs_function)
        if fun == "draw_polyline":
            self.canvas = core.visualization.draw_polyline(self.canvas, **kwargs_function)
        if fun == "draw_reference":
            self.canvas = core.visualization.draw_reference(self.canvas, **kwargs_function)

        ## export
        if fun == "convert_annotation":
            annotations_updated = core.export.convert_annotation(**kwargs_function)
        if fun == "save_annotation":
            if not "file_name" in kwargs_function:
                kwargs_function["file_name"] = self._construct_file_name("annotations", "json")
            core.export.save_annotation(dir_path=self.dir_path,**kwargs_function)
        if fun == "save_canvas":
            if not "file_name" in kwargs_function:
                ext = kwargs_function.get("ext", ".jpg")
                kwargs_function["file_name"] = self._construct_file_name("canvas", ext)
            core.export.save_canvas(
                self.canvas,
                dir_path=self.dir_path,
                **kwargs_function,
            )
        if fun == "save_ROI":
            if not "file_name" in kwargs_function:
                ext = kwargs_function.get("ext", ".jpg")
                kwargs_function["file_name"] = self._construct_file_name("canvas", ext)
            core.export.save_ROI(
                self.image_copy,
                dir_path=self.dir_path,
                **kwargs_function,
            )

        if fun == "export_csv":
            core.export.export_csv(
                dir_path=self.dir_path,
                save_prefix=self.file_prefix,
                save_suffix=self.file_suffix,
                image_name=image_name,
                **kwargs_function,
            )

        ## save annotation to dict
        if annotations_updated:

            if not annotation_type in annotations:
                annotations[annotation_type] = {}

            annotations[annotation_type][annotation_id] = annotations_updated[annotation_type][annotation_id]
            self.annotations.update(annotations)


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
        flag_autosave = False

        ## check dir_path
        print("\nAUTOSAVE")

        if hasattr(self, "canvas") and not "save_canvas" in export_list:
            print("- save_canvas")
            core.export.save_canvas(
                self.canvas,
                file_name=self._construct_file_name("canvas", "jpg"),
                dir_path=self.dir_path,
                **kwargs,
            )
            flag_autosave = True

        if hasattr(self, "annotations") and not "save_annotation" in export_list:
            print("- save_annotation")
            core.export.save_annotation(
                self.annotations,
                file_name=self._construct_file_name("annotations", "json"),
                dir_path=self.dir_path,
                **kwargs,
            )
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




def load_image(path, mode="default", **kwargs):
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
    flags = make_dataclass(cls_name="flags", fields=[("mode", str, mode)])

    ## load image
    if path.__class__.__name__ == "str":
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext.replace(".", "") in settings.default_filetypes:
                if flags.mode == "default":
                    image = cv2.imread(path)
                elif flags.mode == "colour":
                    image = cv2.imread(path, cv2.IMREAD_COLOR)
                elif flags.mode == "gray":
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                elif flags.mode == "rgb":
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                print(
                    'Invalid file extension "{}" - could not load image:\n'.format(ext)
                )
                return
        elif os.path.isdir(path):
            image = utils_lowlevel._load_project_image_directory(
                path, as_container=False
            )
        else:
            print("Invalid image path - could not load image.")
            return
    else:
        print("Invalid input format - could not load image.")
        return

    return image


def load_template(
    template_path,
    tag="v1",
    overwrite=False,
    keep_comments=True,
    image_path=None,
    dir_path=None,
    ret_path=False,
):
    """

    Parameters
    ----------
    template_path : TYPE
        DESCRIPTION.
    tag : TYPE, optional
        DESCRIPTION. The default is "v1".
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.
    keep_comments : TYPE, optional
        DESCRIPTION. The default is True.
    image_path : TYPE, optional
        DESCRIPTION. The default is None.
    dir_path : TYPE, optional
        DESCRIPTION. The default is None.
    ret_path : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    config_path : TYPE
        DESCRIPTION.

    """

    flags = make_dataclass(cls_name="flags", fields=[("overwrite", bool, overwrite)])

    ## create config from template
    if template_path.__class__.__name__ == "str":
        if os.path.isfile(template_path):
            template_loaded = utils_lowlevel._load_yaml(template_path)
        else:
            print("Could not find template_path")
            return
    else:
        print("Wrong input format for template_path")
        return

    ## construct config-name
    if (
        dir_path.__class__.__name__ == "NoneType"
        and image_path.__class__.__name__ == "NoneType"
    ):
        print("Need to specify image_path or dir_path")
        return

    elif (
        dir_path.__class__.__name__ == "str"
        and image_path.__class__.__name__ == "NoneType"
    ):
        if os.path.isdir(dir_path):
            prepend = ""
        else:
            print("Could not find dir_path")
            return

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
    if "template_locked" in template_loaded:
        template_loaded.pop("template_locked")

    config_info = {
        "config_info": {
            "config_name": config_name,
            "date_created": datetime.today().strftime(settings.strftime_format),
            "date_last_modified": None,
            "template_name": os.path.basename(template_path),
            "template_path": template_path,
        }
    }

    yaml = ruamel.yaml.YAML()
    yaml.width = 4096
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

    if utils_lowlevel._save_prompt("template", config_path, flags.overwrite):
        with open(config_path, "wb") as yaml_file:
            yaml.dump(template_loaded, yaml_file)

    if ret_path:
        return config_path



def print_colours():

    colours_path = os.path.join(resource_filename("phenopype", "assets"), "wc3_colours.html")
    webbrowser.open_new_tab(colours_path)





def save_image(
    image,
    file_name,
    dir_path,
    resize=1,
    ext="jpg",
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
    if file_name[len(file_name)-4] == ".":
        new_name = file_name
    elif file_name.endswith(ext):
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
            print_msg = "- image saved under " + path + " (overwritten)."
            pass
        elif not os.path.isfile(path):
            print_msg = "- image saved under " + path + "."
            pass

        if settings.flag_verbose:
            print(print_msg)

        cv2.imwrite(path, image)
        break


def show_image(
    image,
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
                    utils_lowlevel._GUI(
                        i,
                        mode="",
                        window_aspect=window_aspect,
                        window_name="phenopype" + " - " + str(idx),
                        window_control="external",
                        **kwargs,
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
            utils_lowlevel._GUI(
                image=image,
                mode="",
                window_aspect=window_aspect,
                window_name="phenopype",
                window_control="internal",
                **kwargs,
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
