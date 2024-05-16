#%% modules

import copy
import inspect
import io
import json
import logging
import os
import re
import ruamel.yaml
import shutil
import string
import sys
import warnings

from _ctypes import PyObj_FromPtr
from contextlib import redirect_stdout
from dataclasses import dataclass, fields, make_dataclass
from math import atan2, cos, pi, sin, sqrt
from pathlib import Path
from stat import S_IWRITE
from timeit import default_timer as timer

import cv2
import numpy as np
from colour import Color
from PIL import Image
from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


from phenopype import _vars
from phenopype import config
from phenopype import core
from phenopype import utils

try:
    import phenopype_plugins as plugins
except:
    pass

#%% classes

class _Container(object):
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

    def _load(self, contours=False, **kwargs):
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
            try:
                annotations_loaded = core.export.load_annotation(
                    os.path.join(self.dir_path, annotations_file_name)
                )
                
                if contours == False:
                    if _vars._contour_type in annotations_loaded:
                        annotations_loaded.pop(_vars._contour_type)
                
                if annotations_loaded:
                    self.annotations.update(annotations_loaded)
    
                annotation_types_loaded = {}
                for annotation_type in self.annotations.keys():
                    id_list = []
                    for annotation_id in self.annotations[annotation_type].keys():
                        id_list.append(annotation_id)
                    if len(id_list) > 0:
                        annotation_types_loaded[annotation_type] = _NoIndent(
                            id_list
                        )
    
                loaded.append(
                    "- annotations loaded:\n{}".format(
                        json.dumps(
                            annotation_types_loaded,
                            indent=4,
                            cls=_NoIndentEncoder,
                        ).replace("}", "").replace("{", "")
                    )
                )
            except:
                print("WARNING - BROKEN ANNOTATIONS FILE")

        ## load global objects from project attributes
        self.attr_proj = _load_yaml(os.path.join(self.dir_path, r"../../", "attributes.yaml"), typ="safe")
        if "reference_templates" in self.attr_proj:
            for reference_id, reference_info in self.attr_proj["reference_templates"].items():
                if not reference_id in config.reference_templates:
                    config.reference_templates[reference_id] = reference_info
                else:
                    config.reference_templates[reference_id].update(reference_info)    
            loaded.append("loaded info for {} reference templates {} ".format(len(config.reference_templates.keys()),(*list(config.reference_templates.keys()),)))                
        if "models" in self.attr_proj:          
            for model_id, model_info in self.attr_proj["models"].items():
                if not model_id in config.models:
                    config.models[model_id] = model_info
                else:
                    config.models[model_id].update(model_info)    
            loaded.append("loaded info for {} models {} ".format(len(config.models.keys()),(*list(config.models.keys()),)))
            
        ## feedback
        if len(loaded) > 0:
            print("\n- ".join(loaded))
        else:
            print("- nothing to autoload")

    def _reset(self):
        """
        Resets modified images, canvas and df_image_data to original state. Can be used manually, but is typically used within the
        pype routine.

        """

        ## re-assign copies
        self.image = copy.deepcopy(self.image_copy)
        self.canvas = copy.deepcopy(self.image_copy)

    def _run(
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
        if config.verbose:
            kwargs_function["verbose"] = True

        ## indicate pype use 
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
                            kwargs_function["interactive"] = False
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
        if fun == "clip_histogram":
            self.image = core.preprocessing.clip_histogram(self.image, **kwargs_function)
        if fun == "create_mask":
            annotations_updated = core.preprocessing.create_mask(self.image, **kwargs_function)
        if fun == "create_reference":
            annotations_updated = core.preprocessing.create_reference(self.image, **kwargs_function)
        if fun == "detect_mask":
            annotations_updated = core.preprocessing.detect_mask(self.image, **kwargs_function)
        if fun == "detect_QRcode":
            annotations_updated = core.preprocessing.detect_QRcode(self.image, **kwargs_function)
        if fun == "write_comment":
            annotations_updated = core.preprocessing.write_comment(self.image, **kwargs_function)
        if fun == "detect_reference":
            template_id = kwargs_function["template_id"]
            if "template" not in config.reference_templates[template_id]:
                print(f"- loading reference template \"{template_id}\" into memory")
                config.reference_templates[template_id]["template"] = utils.load_image(
                    config.reference_templates[template_id]["template_path"])
            annotations_updated = core.preprocessing.detect_reference(
                self.image,
                config.reference_templates[template_id]["template"],
                config.reference_templates[template_id]["template_px_ratio"],
                config.reference_templates[template_id]["unit"],
                **kwargs_function)

        if fun == "decompose_image":
            self.image = core.preprocessing.decompose_image(self.image_copy, **kwargs_function)
        if fun == "manage_channels":
            self.image_channels = core.preprocessing.manage_channels(self.image, **kwargs_function)

        ## core.segmentation
        if fun == "contour_to_mask":
            annotations_updated = core.segmentation.contour_to_mask(**kwargs_function)
        if fun == "threshold":
            self.image = core.segmentation.threshold(self.image, **kwargs_function)
        if fun == "watershed":
            self.image = core.segmentation.watershed(self.image, **kwargs_function)
        if fun == "mask_to_contour":
            annotations_updated = core.segmentation.mask_to_contour(**kwargs_function)
        if fun == "morphology":
            self.image = core.segmentation.morphology(self.image, **kwargs_function)
        if fun == "detect_contour":
            annotations_updated = core.segmentation.detect_contour(self.image, **kwargs_function)
        if fun == "edit_contour":
            annotations_updated, self.image = core.segmentation.edit_contour(
                self.canvas, ret_image=True, **kwargs_function
            )
            if "inplace" in kwargs_function:
                annotations[_vars._contour_type][kwargs_function["contour_id"]] = core.segmentation.detect_contour(self.image)[_vars._contour_type]["a"]
                self.image = copy.deepcopy(self.image_copy)
                # self.annotations.update(annotations)

        ## plugins.segmentation
        if fun == "predict_fastSAM":
            self.image = plugins.segmentation.predict_fastSAM(self.image_copy, **kwargs_function)
        if fun == "predict_keras":
            self.image = plugins.segmentation.predict_keras(self.image_copy,  **kwargs_function)
        if fun == "predict_torch":
            self.image = plugins.segmentation.predict_torch(self.image_copy, **kwargs_function)

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
                self.image, **kwargs_function
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
        if fun == "draw_comment":
            self.canvas = core.visualization.draw_comment(self.canvas, **kwargs_function)
        if fun == "draw_contour":
            self.canvas = core.visualization.draw_contour(self.canvas, **kwargs_function)
        if fun == "draw_landmark":
            self.canvas = core.visualization.draw_landmark(self.canvas, **kwargs_function)
        if fun == "draw_mask":
            self.canvas = core.visualization.draw_mask(self.canvas, **kwargs_function)
        if fun == "draw_polyline":
            self.canvas = core.visualization.draw_polyline(self.canvas, **kwargs_function)
        if fun == "draw_QRcode":
            self.canvas = core.visualization.draw_QRcode(self.canvas, **kwargs_function)
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
                kwargs_function["file_name"] = self._construct_file_name("roi", ext)
            if not "dir_path" in kwargs_function:
                kwargs_function["dir_path"] = os.path.join(self.dir_path, "ROI")
                if not os.path.isdir(kwargs_function["dir_path"]):
                    os.makedirs(kwargs_function["dir_path"])
            core.export.save_ROI(
                self.image_copy,
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


    def _save(self, dir_path=None, export_list=[], overwrite=False, **kwargs):
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


@dataclass
class _GUI_Settings:
    """
    Configuration settings for the _GUI class, defining various parameters for GUI behavior and appearance.
    
    Attributes:
    ----------
    comment_key : chr, optional
        Default key binding for adding comments in the GUI. Defaults to None.
    interactive : bool
        Enables or disables interactive features in the GUI. Defaults to True.
    label_colour : tuple
        Default color for labels, specified as a color name or RGB tuple. Defaults to "default", which uses a system-defined color.
    label_size : int
        Default font size for labels. Set to "auto" to automatically adjust based on the GUI size.
    label_width : int
        Line thickness of label text. Set to "auto" for automatic adjustment.
    line_colour : tuple
        Default line color for drawing. Defaults to "default".
    line_width : int
        Thickness of drawn lines. Defaults to "auto" for automatic adjustment.
    node_colour : tuple
        Color for interactive nodes. Defaults to "default".
    node_size : int
        Size of the nodes used in drawings. Defaults to "auto".
    overlay_blend : float
        Opacity for overlay elements. Defaults to 0.2.
    overlay_colour_left : tuple
        Color for the left side of the overlay. Defaults to "default".
    overlay_colour_right : tuple
        Color for the right side of the overlay. Defaults to "default".
    point_colour : tuple
        Default color for points. Defaults to "default".
    point_size : int
        Size of points. Defaults to "auto".
    pype_mode : bool
        Special mode for pipelined operations. Defaults to False.
    return_input : bool
        If True, the GUI returns user inputs for further processing. Defaults to False.
    show_label : bool
        Controls visibility of labels. Defaults to False.
    show_nodes : bool
        Determines whether nodes are visible on the GUI. Defaults to False.
    wait_time : int
        Time in milliseconds the GUI waits for a user input before updating. Defaults to 500.
    window_aspect : str
        The aspect ratio of the GUI window. Defaults to "normal".
    window_control : str
        Controls how the window should behave; either 'internal' or a custom specification. Defaults to "internal".
    window_name : str
        The title of the GUI window. Defaults to "phenopype".
    zoom_magnification : float
        Magnification factor for zoom. Defaults to 0.5.
    zoom_memory : bool
        If True, remembers the last zoom state between sessions. Defaults to False.
    zoom_mode : str
        The mode of zooming, such as 'continuous' or 'step'. Defaults to "continuous".
    zoom_n_steps : int
        Number of steps for zoom adjustment. Defaults to 20.
    """
    comment_key: chr = None
    interactive: bool = True
    label_colour: tuple = "default"
    label_size: int = "auto"
    label_width: int = "auto"
    line_colour: tuple = "default"
    line_width: int = "auto"
    node_colour: tuple = "default"
    node_size: int = "auto"
    overlay_blend: float = 0.2
    overlay_colour_left: tuple = "default"
    overlay_colour_right: tuple = "default"
    point_colour: tuple = "default"
    point_size: int = "auto"
    pype_mode: bool = False
    return_input: bool = False
    show_label: bool = False
    show_nodes: bool = False
    wait_time: int = 500
    window_aspect: str = "normal"
    window_control: str = "internal"
    window_name: str = "phenopype"
    zoom_magnification: float = 0.5
    zoom_memory: bool = False
    zoom_mode: str = "continuous"
    zoom_n_steps: int = 20

class _GUI:
    def __init__(
        self,
        image,
        tool=None,
        interactive=True,
        **kwargs
    ):
        """
        Central class to manage the Graphical User Interface (GUI) for user-image interaction in phenopype.
    
        This class initializes and manages an image display and interaction system using OpenCV windows. It supports various
        tools and settings to interact dynamically with images for purposes such as annotation, labeling, and image manipulation.
        It handles image resizing, tool initialization, and user input to perform tasks like drawing, commenting, and applying
        transformations. See the _GUI_Settings dataclass documentation for a list of keyword arguments.
    
        Parameters:
        ----------
        image : ndarray
            The image array (numpy array) on which the GUI operations will be performed.
        tool : optional, default=None
            The initial tool to load into the GUI for image manipulation. This can be a string identifier like 'draw', 'comment', etc.
        interactive : bool, optional, default=True
            Specifies whether the GUI should be interactive. If set to True, user input is enabled and GUI updates respond to interactions.
        **kwargs : dict
            Additional keyword arguments that may specify further GUI settings or override default settings for aspects like window dimensions,
            color and size of graphical elements - see the _GUI_Settings dataclass documentation for a list of keyword arguments.

        """
        
        ## configure image   
        if not image.__class__.__name__ == "ndarray":
            raise TypeError("GUI module did not receive array-type - aborting!")
        window_max_dim = kwargs.get("window_max_dim", config.window_max_dim)
        window_min_dim = kwargs.get("window_min_dim", config.window_min_dim)
        
        ## get canvas dimensions
        self._prepare_canvas(image, window_max_dim, window_min_dim)     

        ## load settings from settings class
        self._initialize_settings(tool, interactive, kwargs)
        
        ## initialize canvas (load previous zoom, mount, etc.)
        self._initialize_canvas()     
        
        ## prepare GUI for the evebtual use of certain tools (comment and drawing)
        self._prepare_tools(kwargs)
        



        ## RUN: load tools and allow for user input
        if self.settings.interactive:

            cv2.namedWindow(
                self.settings.window_name, _vars.opencv_window_flags[self.settings.window_aspect]
            )
            cv2.startWindowThread()
            cv2.setMouseCallback(self.settings.window_name, self._on_mouse_plain)
            cv2.resizeWindow(
                self.settings.window_name, self.canvas_width, self.canvas_height
            )
            cv2.imshow(self.settings.window_name, self.canvas)
            # cv2.setWindowProperty(self.settings.window_name, cv2.WND_PROP_TOPMOST, 1)
            self.keypress = None

            if self.settings.window_control == "internal":
                while not any([self.flags.end, self.flags.end_pype]):
                        
                    ## sync zoom settings with config
                    config.gui_zoom_config = self.zoom

                    ## directly return key input
                    if self.settings.return_input:
                        self.keypress = cv2.waitKey(0)
                        self._keyboard_input()
                        self.flags.end = True
                        cv2.destroyAllWindows()

                    ## comment tool
                    if self.tool == "comment":
                        self.keypress = cv2.waitKey(1)
                        self._comment_tool()
                    elif self.tool == "labelling":
                        self.keypress = cv2.waitKeyEx(0)
                        self._labelling_tool()
                    else:
                        self.keypress = cv2.waitKey(self.settings.wait_time)
                        
                    ## draw nodes
                    if self.tool in ["rectangle", "polygon", "polyline"]:
                        if self.settings.show_nodes:
                            for coord_list in self.data[_vars._coord_list_type]:
                                self._canvas_draw(
                                    tool="point", 
                                    coord_list=coord_list,
                                    size=self.settings.node_size,
                                    colour=self.settings.node_colour,
                                    )
                        if self.flags.finished and kwargs.get("labelling"):
                            cv2.waitKeyEx(self.settings.wait_time)
                            self.flags.end = True

                    ## Enter = close window and redo
                    if self.keypress == 13:
                        ## close unfinished polygon and append to polygon list
                        if self.tool:
                            if len(
                                self.data[_vars._coord_type]
                            ) > 2 and not self.tool in ["point"]:
                                if not self.tool in ["polyline"]:
                                    self.data[_vars._coord_type].append(
                                        self.data[_vars._coord_type][0]
                                    )
                                self.data[_vars._coord_list_type].append(
                                    self.data[_vars._coord_type]
                                )
                        self.flags.end = True
                        cv2.destroyAllWindows()

                    ## Ctrl + Enter = close window and move on
                    elif self.keypress == 10:
                        self.flags.end = True
                        self.flags.end_pype = True
                        cv2.destroyAllWindows()

                    ## Esc = close window and terminate
                    elif self.keypress == 27 and not kwargs.get("labelling"):
                        cv2.destroyAllWindows()
                        logging.shutdown()
                        sys.exit("\n\nTERMINATE (by user)")

                    ## Ctrl + z = undo
                    elif self.keypress == 26 and self.tool == "draw":
                        self.data[_vars._sequence_type] = self.data[
                            _vars._sequence_type
                        ][:-1]
                        self._canvas_renew()
                        self._canvas_draw(
                            tool="line_bin",
                            coord_list=self.data[_vars._sequence_type],
                        )
                        self._canvas_blend()
                        self._canvas_draw_contours()
                        self._canvas_mount()

                    ## external window close
                    elif config.window_close:
                        self.flags.end = True
                        cv2.destroyAllWindows()
                        
                    if kwargs.get("labelling"):
                        if self.keypress in [13, 27, 2424832, 2555904]:
                            self.flags.end = True
                            cv2.destroyAllWindows()
                            
        ## RUN: finish without loading tools or allowing feedback
        else:
            self.flags.end = True
            self.flags.end_pype = True
                                
                      
    def _prepare_canvas(self, image, window_max_dim, window_min_dim):
        """
        Adjust the canvas size based on maximum and minimum dimension constraints while maintaining aspect ratio.
    
        Args:
            window_max_dim (int): The maximum allowed dimension for either width or height.
            window_min_dim (int): The minimum allowed dimension for either width or height.
        """
        self.image = copy.deepcopy(image)
        self.image_width, self.image_height = self.image.shape[1], self.image.shape[0]
        aspect_ratio = self.image_width / self.image_height
    
        # Determine the dimension constraints based on the aspect ratio
        if self.image_width >= self.image_height:
            # Width is the dominating dimension or they are equal
            target_width = min(window_max_dim, max(self.image_width, window_min_dim))
            target_height = int(target_width / aspect_ratio)
        else:
            # Height is the dominating dimension
            target_height = min(window_max_dim, max(self.image_height, window_min_dim))
            target_width = int(target_height * aspect_ratio)
    
        # Apply constraints to not exceed max dimensions and not fall below min dimensions
        self.canvas_width = min(target_width, window_max_dim)
        self.canvas_height = min(target_height, window_max_dim)
        
        # Ensure the canvas dimensions do not fall below the minimum dimension constraints
        if self.canvas_width < window_min_dim:
            self.canvas_width = window_min_dim
            self.canvas_height = int(self.canvas_width / aspect_ratio)
        if self.canvas_height < window_min_dim:
            self.canvas_height = window_min_dim
            self.canvas_width = int(self.canvas_height * aspect_ratio)
            
                        
    def _initialize_settings(self, tool, interactive, kwargs):
    
        ## apply args to settings
        self.settings = _GUI_Settings()
        self.settings.tool = tool
        self.settings.interactive = interactive
                        
        ## apply kwargs to setting
        for field in fields(self.settings):
            if field.name in kwargs:
                field_val = kwargs[field.name] 
                setattr(self.settings, field.name, field_val)
            else:
                field_val = field.default
            if "colour" in field.name: 
                field_val = _get_bgr(field_val, field.name)
                setattr(self.settings, field.name, field_val)
            if "size" in field.name or "width" in field.name: 
                field_val = _get_size(self.canvas_height, self.canvas_width, field.name, field_val)
                setattr(self.settings, field.name, field_val)
                 
        ## basic settings (maybe integrate better)
        self.__dict__.update(kwargs)
        self.tool = tool
        self.query = kwargs.get("query", None)

        ## data collector
        self.data = {
            _vars._comment_type: "",
            _vars._contour_type: [],
            _vars._coord_type: [],
            _vars._coord_list_type: [],
            _vars._sequence_type: [],
        }

        self.data.update(kwargs.get("data", {}))

        ## hack to fix empty list bug
        if (type(self.data[_vars._comment_type]) == list and len(self.data[_vars._comment_type]) == 0):
            self.data[_vars._comment_type] = ""

        ## collect interactions and set flags
        self.line_width_orig = copy.deepcopy(self.settings.line_width)

        self.flags = make_dataclass(
            cls_name="flags",
            fields=[
                ("end", bool, False),
                ("end_pype", bool, False),
                ("drawing", bool, False),
                ("rect_start", tuple, None),
                ("finished", bool, False),
                ("comment", bool, False),
            ],
        )
               

    def _initialize_canvas(self):
        
        ## canvas resize factor
        self.canvas_fx, self.canvas_fy = (
            self.image_width / self.canvas_width,
            self.image_height / self.canvas_height,
        )
        
        ## zoom config
        self.zoom = make_dataclass(cls_name="zoom_config", fields=[])
        self.zoom.global_fx, self.zoom.global_fy = self.canvas_fx, self.canvas_fy
        self.zoom.x1, self.zoom.y1, self.zoom.x2, self.zoom.y2 = (
            0,
            0,
            self.image_width,
            self.image_height,
        )
        self.zoom.flag, self.zoom.idx = -1, 1
        self.zoom.step_x, self.zoom.step_y = (
            int(self.image_width / self.settings.zoom_n_steps),
            int(self.image_height / self.settings.zoom_n_steps),
        )
        if self.settings.zoom_mode == "fixed":
            mag = int(self.settings.zoom_magnification * self.settings.zoom_n_steps)
            self.zoom.step_x, self.zoom.step_y = (
                mag * self.zoom.step_x,
                mag * self.zoom.step_y,
            )
        ## update zoom from previous call
        if hasattr(config, "gui_zoom_config") and self.settings.zoom_memory == True:
            if not config.gui_zoom_config.__class__.__name__ == "NoneType":
                self.zoom = config.gui_zoom_config
    
        ## initialize canvas
        self._canvas_renew()

        ## applies zooming step
        self._canvas_mount()
    
        ## local control vars
        config.window_close = False
        
    def _prepare_tools(self, kwargs):
        
        if self.tool in ["comment", "labelling"]:
            
            self.settings.label_keymap = kwargs.get("label_keymap")
            self.settings.label_position = kwargs.get("label_position", (0.1,0.1))
            y_pos, x_pos = self.settings.label_position
            self.canvas = copy.deepcopy(self.canvas_copy)                      
            cv2.putText(
                self.canvas,
                str(self.query) + ": " + str(self.data[_vars._comment_type]),
                (int(self.canvas.shape[0] * y_pos), int(self.canvas.shape[1] * x_pos)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.settings.label_size,
                self.settings.label_colour,
                self.settings.label_width,
                cv2.LINE_AA,
            )

        if self.tool == "draw":
            if len(self.data[_vars._contour_type]) > 0:
                if len(self.image.shape) == 2:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
                self.image_bin = np.zeros(self.image.shape[0:2], dtype=np.uint8)
                self.image_bin_copy = copy.deepcopy(self.image_bin)
                for contour in self.data[_vars._contour_type]:
                    cv2.drawContours(
                        image=self.image_bin,
                        contours=[contour],
                        contourIdx=0,
                        thickness=-1,
                        color=255,
                        maxLevel=3,
                        offset=(0, 0),
                    )
                self._canvas_draw(
                    tool="line_bin", 
                    coord_list=self.data[_vars._sequence_type],
                )
                self._canvas_blend()
                self._canvas_draw_contours()
                    
            else:
                print("Could not find contours to edit - check annotations.")
                return
            
        if self.tool in ["rectangle", "polygon", "polyline", "draw"]:
            self._canvas_draw(
                tool="line", 
                coord_list=self.data[_vars._coord_list_type],
                colour=self.settings.line_colour, 
                width=self.settings.line_width,
                )
            if self.settings.show_nodes:
                for coord_list in self.data[_vars._coord_list_type]:
                    self._canvas_draw(
                        tool="point", 
                        coord_list=coord_list,
                        size=self.settings.node_size,
                        colour=self.settings.node_colour,
                        )
        if self.tool in ["point"]:
            self._canvas_draw(
                tool="point", 
                coord_list=self.data[_vars._coord_type],
                size=self.settings.point_size,
                colour=self.settings.point_colour,
                )

            
     
    def _keyboard_input(self):
        self.keypress_trans = chr(self.keypress)
        return self

    def _comment_tool(self):

        if self.keypress > 0 and not self.keypress in [8, 13, 27]:
            self.data[_vars._comment_type] = self.data[_vars._comment_type] + chr(
                self.keypress
            )
        elif self.keypress == 8:
            self.data[_vars._comment_type] = self.data[_vars._comment_type][
                0 : len(self.data[_vars._comment_type]) - 1
            ]

        self.canvas = copy.deepcopy(self.canvas_copy)
        y_pos, x_pos = self.settings.label_position

        cv2.putText(
            self.canvas,
            str(self.query) + ": " + str(self.data[_vars._comment_type]),
            (int(self.canvas.shape[0] * y_pos), int(self.canvas.shape[1] * x_pos)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.settings.label_size,
            self.settings.label_colour,
            self.settings.label_width,
            cv2.LINE_AA,
        )
        cv2.imshow(self.settings.window_name, self.canvas)
        
        
    def _labelling_tool(self):
        
        y_pos, x_pos = self.settings.label_position
        
        if self.keypress in [13, 27, 2424832, 2555904]:
            self.flags.end = True
        elif self.keypress in _vars.ascii_codes:
            key = str(_vars.ascii_codes[self.keypress])
            if key in self.settings.label_keymap:
                self.data[_vars._comment_type] = str(self.settings.label_keymap[_vars.ascii_codes[self.keypress]])
                self.canvas = copy.deepcopy(self.canvas_copy)
                cv2.putText(
                    self.canvas,
                    str(self.query) + ": " + self.data[_vars._comment_type],
                    (int(self.canvas.shape[0] * y_pos), int(self.canvas.shape[1] * x_pos)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.settings.label_size,
                    self.settings.label_colour,
                    self.settings.label_width,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.settings.window_name, self.canvas)
                cv2.waitKeyEx(self.settings.wait_time)
                self.flags.end = True
            elif key == self.settings.comment_key:
                self.flags.comment=True
                print("COMMENTING!")
            else:
                print(f"key {key} not coded!")
        else:
            print(f"{self.keypress} is not a valid ASCII code")
# 
    def _on_mouse_plain(self, event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEWHEEL and not self.keypress == 9:
            self.keypress = None
            if flags > 0:
                if self.zoom.idx < self.settings.zoom_n_steps:
                    self.zoom.flag = 1
                    self.zoom.idx += 1
                    if self.settings.zoom_mode == "continuous" or (
                        self.settings.zoom_mode == "fixed" and self.zoom.idx == 2
                    ):
                        self._zoom_fun(x, y)
            if flags < 0:
                if self.zoom.idx > 1:
                    self.zoom.flag = -1
                    self.zoom.idx -= 1
                    if self.settings.zoom_mode == "continuous" or (
                        self.settings.zoom_mode == "fixed" and self.zoom.idx == 1
                    ):
                        self._zoom_fun(x, y)
            self.x, self.y = x, y
            cv2.imshow(self.settings.window_name, self.canvas)

        if self.tool:
            if self.tool == "draw":
                self._on_mouse_draw(event, x, y, flags)
            elif self.tool == "point":
                self._on_mouse_point(event, x, y)
            elif self.tool == "polygon":
                self._on_mouse_polygon(event, x, y, flags)
            elif self.tool == "polyline" or self.tool == "polylines":
                self._on_mouse_polygon(event, x, y, flags, polyline=True)
            elif self.tool == "rectangle":
                self._on_mouse_rectangle(event, x, y, flags)
            elif self.tool == "reference":
                self._on_mouse_polygon(event, x, y, flags, reference=True)
            elif self.tool == "template":
                self._on_mouse_rectangle(event, x, y, flags, template=True)

    def _on_mouse_point(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:

            ## convert cursor coords from zoomed canvas to original coordinate space
            self._zoom_coords_orig(x, y)

            ## append points to point list
            self.data[_vars._coord_type].append(self.coords_original)

            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="point", 
                coord_list=self.data[_vars._coord_type],
                size=self.settings.point_size,
                colour=self.settings.point_colour,
                )
            self._canvas_mount()

        if event == cv2.EVENT_RBUTTONDOWN:

            ## remove points from list, if any are left
            if len(self.data[_vars._coord_type]) > 0:
                self.data[_vars._coord_type] = self.data[_vars._coord_type][:-1]

            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="point", 
                coord_list=self.data[_vars._coord_type],
                size=self.settings.point_size,
                colour=self.settings.point_colour,
                )
            self._canvas_mount()

    def _on_mouse_polygon(self, event, x, y, flags, **kwargs):

        ## kwargs
        polyline = kwargs.get("polyline", False)
        reference = kwargs.get("reference", False)
        flag_draw = kwargs.get("draw", False)

        if event == cv2.EVENT_MOUSEMOVE:
            if (
                (reference or flag_draw)
                and len(self.data[_vars._coord_type]) == 2
            ):
                return

            ## draw line between current cursor coords and last polygon node
            if len(self.data[_vars._coord_type]) > 0:
                self.coords_prev = (
                    int(
                        (self.data[_vars._coord_type][-1][0] - self.zoom.x1)
                        / self.zoom.global_fx
                    ),
                    int(
                        (self.data[_vars._coord_type][-1][1] - self.zoom.y1)
                        // self.zoom.global_fy
                    ),
                )
                self.canvas = copy.deepcopy(self.canvas_copy)
                cv2.line(
                    self.canvas,
                    self.coords_prev,
                    (x, y),
                    self.settings.line_colour,
                    self.settings.line_width,
                )
                if self.settings.show_nodes:
                    cv2.circle(
                        self.canvas,
                        self.coords_prev,
                        self.settings.node_size,
                        self.settings.node_colour,
                        -1,
                        )
            ## if in reference mode, don't connect
            elif (
                (reference or flag_draw)
                and self.tool == "line"
                and len(self.data[_vars._coord_type]) > 2
            ):
                pass

            ## pump updates
            cv2.imshow(self.settings.window_name, self.canvas)

        if event == cv2.EVENT_LBUTTONDOWN:

            ## skip if in reference mode
            if reference and len(self.data[_vars._coord_type]) == 2:
                print("already two points selected")
                return

            ## convert cursor coords from zoomed canvas to original coordinate space
            self._zoom_coords_orig(x, y)

            ## append points to point list
            self.data[_vars._coord_type].append(self.coords_original)

            ## apply tool and refresh canvas
            self._canvas_renew()            
            self._canvas_draw(
                tool="line", 
                coord_list=self.data[_vars._coord_list_type] + [self.data[_vars._coord_type]],
                colour=self.settings.line_colour, 
                width=self.settings.line_width,
                )
            
            if self.settings.show_nodes:
                for coord_list in self.data[_vars._coord_list_type] + [self.data[_vars._coord_type]]:
                    self._canvas_draw(
                        tool="point", 
                        coord_list=coord_list,
                        size=self.settings.node_size,
                        colour=self.settings.node_colour,
                        )
            self._canvas_mount()

            ## if in reference mode, append to ref coords
            if reference and len(self.data[_vars._coord_type]) == 2:
                print("Reference set")

        if event == cv2.EVENT_RBUTTONDOWN:

            ## remove points and update canvas
            if len(self.data[_vars._coord_type]) > 0:
                self.data[_vars._coord_type] = self.data[_vars._coord_type][:-1]
            else:
                self.data[_vars._coord_list_type] = self.data[ _vars._coord_list_type][:-1]

            ## apply tool and refresh canvas
            print("remove")
            self._canvas_renew()
            self._canvas_draw(
                tool="line", 
                coord_list=self.data[_vars._coord_list_type] + [self.data[_vars._coord_type]],
                colour=self.settings.line_colour, 
                width=self.settings.line_width,
                )
            if self.settings.show_nodes:
                for coord_list in self.data[_vars._coord_list_type] + [self.data[_vars._coord_type]]:
                    self._canvas_draw(
                        tool="point", 
                        coord_list=coord_list,
                        size=self.settings.node_size,
                        colour=self.settings.node_colour,
                        )
            self._canvas_mount()

        if flags == cv2.EVENT_FLAG_CTRLKEY and len(self.data[_vars._coord_type]) >= 2:

            ## close polygon
            if not polyline:
                self.data[_vars._coord_type].append(
                    self.data[_vars._coord_type][0]
                )

            ## add current points to polygon and empyt point list
            print("poly")
            self.data[_vars._coord_list_type].append(self.data[_vars._coord_type])
            self.data[_vars._coord_type] = []

            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="line", 
                coord_list=self.data[_vars._coord_list_type] + [self.data[_vars._coord_type]],
                colour=self.settings.line_colour, 
                width=self.settings.line_width,
                )
            self._canvas_mount()
            self.flags.finished = True

    def _on_mouse_rectangle(self, event, x, y, flags, **kwargs):

        ## kwargs
        template = kwargs.get("template", False)

        if event == cv2.EVENT_LBUTTONDOWN:

            ## end after one set of points if creating a template
            if template == True and len(self.data[_vars._coord_list_type]) == 1:
                return

            ## start drawing temporary rectangle
            self.flags.rect_start = x, y
            self.canvas_copy = copy.deepcopy(self.canvas)
            
            if self.settings.show_nodes:
                for coord_list in self.data[_vars._coord_list_type]:
                    self._canvas_draw(
                        tool="point", 
                        coord_list=coord_list,
                        size=self.settings.node_size,
                        colour=self.settings.node_colour,
                        )

        if event == cv2.EVENT_LBUTTONUP:

            ## end after one set of points if creating a template
            if template == True and len(self.data[_vars._coord_list_type]) == 1:
                print("Template selected")
                return

            ## end drawing temporary rectangle
            self.flags.rect_start = None

            ## convert rectangle to polygon coords
            self.rect = [
                int(self.zoom.x1 + (self.zoom.global_fx * self.rect_minpos[0])),
                int(self.zoom.y1 + (self.zoom.global_fy * self.rect_minpos[1])),
                int(self.zoom.x1 + (self.zoom.global_fx * self.rect_maxpos[0])),
                int(self.zoom.y1 + (self.zoom.global_fy * self.rect_maxpos[1])),
            ]
            self.data[_vars._coord_list_type].append(
                [
                    (self.rect[0], self.rect[1]),
                    (self.rect[2], self.rect[1]),
                    (self.rect[2], self.rect[3]),
                    (self.rect[0], self.rect[3]),
                    (self.rect[0], self.rect[1]),
                ]
            )

            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="line", 
                coord_list=self.data[_vars._coord_list_type],
                colour=self.settings.line_colour, 
                width=self.settings.line_width,
                )
            if self.settings.show_nodes:
                for coord_list in self.data[_vars._coord_list_type]:
                    self._canvas_draw(
                        tool="point", 
                        coord_list=coord_list,
                        size=self.settings.node_size,
                        colour=self.settings.node_colour,
                        )
            self._canvas_mount(refresh=False)
            self.flags.finished = True

        if event == cv2.EVENT_RBUTTONDOWN:

            ## remove polygons and update canvas
            if len(self.data[_vars._coord_list_type]) > 0:
                self.data[_vars._coord_list_type] = self.data[
                    _vars._coord_list_type
                ][:-1]

                ## apply tool and refresh canvas
                self._canvas_renew()
                self._canvas_draw(
                    tool="line", 
                    coord_list=self.data[_vars._coord_list_type],
                    colour=self.settings.line_colour, 
                    width=self.settings.line_width,
                    )
                if self.settings.show_nodes:
                    for coord_list in self.data[_vars._coord_list_type]:
                        self._canvas_draw(
                            tool="point", 
                            coord_list=coord_list,
                            size=self.settings.node_size,
                            colour=self.settings.node_colour,
                            )
                self._canvas_mount()

        ## draw temporary rectangle
        elif self.flags.rect_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.canvas = copy.deepcopy(self.canvas_copy)
                self.rect_minpos = (
                    min(self.flags.rect_start[0], x),
                    min(self.flags.rect_start[1], y),
                )
                self.rect_maxpos = (
                    max(self.flags.rect_start[0], x),
                    max(self.flags.rect_start[1], y),
                )
                cv2.rectangle(
                    self.canvas,
                    self.rect_minpos,
                    self.rect_maxpos,
                    self.settings.line_colour,
                    self.settings.line_width,
                )
                cv2.imshow(self.settings.window_name, self.canvas)

    def _on_mouse_draw(self, event, x, y, flags):

        ## set colour - left/right mouse button use different settings.colours
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.colour_current_bin = 255
                self.colour_current = self.settings.overlay_colour_left
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.colour_current_bin = 0
                self.colour_current = self.settings.overlay_colour_right

            ## start drawing and use current coords as start point
            self.canvas = copy.deepcopy(self.canvas_copy)

            ## convert cursor coords from zoomed canvas to original coordinate space
            self.ix, self.iy = x, y
            self.coords_original_i = (
                int(self.zoom.x1 + (self.ix * self.zoom.global_fx)),
                int(self.zoom.y1 + (self.iy * self.zoom.global_fy)),
            )
            self.data[_vars._coord_type].append(self.coords_original_i)
            self.flags.drawing = True

        ## finish drawing and update image_copy
        if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.flags.drawing = False
            self.canvas = copy.deepcopy(self.canvas_copy)
            self.data[_vars._sequence_type].append(
                [
                    self.data[_vars._coord_type],
                    self.colour_current_bin,
                    int(self.settings.line_width * self.zoom.global_fx),
                ]
            )
            self.data[_vars._coord_type] = []

            ## draw all segments
            self._canvas_renew()
            self._canvas_draw(
                tool="line_bin", coord_list=self.data[_vars._sequence_type]
            )
            self._canvas_blend()
            self._canvas_draw_contours()
            self._canvas_mount()

        ## drawing mode
        elif self.flags.drawing:

            ## convert cursor coords from zoomed canvas to original coordinate space
            self._zoom_coords_orig(x, y)

            ## add points, colour, and line width to point list
            self.data[_vars._coord_type].append(self.coords_original)

            ## draw onto canvas for immediate feedback
            cv2.line(
                self.canvas,
                (self.ix, self.iy),
                (x, y),
                self.colour_current,
                self.settings.line_width,
            )
            self.ix, self.iy = x, y
            cv2.imshow(self.settings.window_name, self.canvas)

        if self.keypress == 9 and event == cv2.EVENT_MOUSEWHEEL:
            if flags > 1:
                self.line_width_orig += 1
            if flags < 1 and self.line_width_orig > 1:
                self.line_width_orig -= 1

            self.canvas = copy.deepcopy(self.canvas_copy)
            self.settings.line_width = int(
                self.line_width_orig
                / ((self.zoom.x2 - self.zoom.x1) / self.image_width)
            )
            cv2.line(
                self.canvas, (x, y), (x, y), _get_bgr("black"), self.settings.line_width
            )
            cv2.line(
                self.canvas,
                (x, y),
                (x, y),
                _get_bgr("white"),
                max(self.settings.line_width - 5, 1),
            )
            cv2.imshow(self.settings.window_name, self.canvas)

    def _canvas_blend(self):

        ## create coloured overlay from binary image
        self.colour_mask = copy.deepcopy(self.image_bin_copy)
        self.colour_mask = cv2.cvtColor(self.colour_mask, cv2.COLOR_GRAY2BGR)
        self.colour_mask[self.image_bin_copy == 0] = self.settings.overlay_colour_right
        self.colour_mask[self.image_bin_copy == 255] = self.settings.overlay_colour_left

        ## blend two canvas layers
        self.image_copy = cv2.addWeighted(
            self.image_copy,
            1 - self.settings.overlay_blend,
            self.colour_mask,
            self.settings.overlay_blend,
            0,
        )

    def _canvas_draw(self, tool, coord_list, colour=None, size=None, width=None):

        ## apply coords to tool and draw on canvas
        for idx, coords in enumerate(coord_list):
            if len(coords) == 0:
                continue
            if tool == "line":
                cv2.polylines(
                    self.image_copy,
                    np.array([coords]),
                    False,
                    colour,
                    width,
                )
            elif tool == "line_bin":
                cv2.polylines(
                    self.image_bin_copy,
                    np.array([coords[0]]),
                    False,
                    coords[1],
                    coords[2],
                )
            elif tool == "point":
                cv2.circle(
                    self.image_copy,
                    tuple(coords),
                    size,
                    colour,
                    -1,
                )
                if self.query:
                    cv2.putText(
                        self.image_copy,
                        str(idx + 1),
                        coords,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.settings.label_size,
                        self.settings.label_colour,
                        self.settings.label_width,
                        cv2.LINE_AA,
                    )
                    
    def _canvas_draw_contours(self):

        self.contours, self.hierarchies = cv2.findContours(
            image=self.image_bin_copy,
            mode=cv2.RETR_CCOMP,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in self.contours:
            cv2.drawContours(
                image=self.image_copy,
                contours=[contour],
                contourIdx=0,
                thickness=self.settings.line_width,
                color=self.settings.overlay_colour_left,
                maxLevel=3,
                offset=None,
            )

    def _canvas_mount(self, refresh=True):

        ## pass zoomed part of original image to canvas
        self.canvas = self.image_copy[
            self.zoom.y1 : self.zoom.y2, self.zoom.x1 : self.zoom.x2
        ]

        ## resize canvas to fit window
        self.canvas = cv2.resize(
            self.canvas,
            (self.canvas_width, self.canvas_height),
            interpolation=cv2.INTER_LINEAR,
        )

        ## copy canvas for mousedrag refresh
        self.canvas_copy = copy.deepcopy(self.canvas)

        ## refresh canvas
        if refresh and self.settings.interactive:
            cv2.imshow(self.settings.window_name, self.canvas)

    def _canvas_renew(self):

        ## pull copy from original image
        self.image_copy = copy.deepcopy(self.image)
        if self.tool == "draw" and hasattr(self, "image_bin"):
            self.image_bin_copy = copy.deepcopy(self.image_bin)

    def _zoom_fun(self, x, y):
        """
        Helper function for image_viewer. Takes current xy coordinates and 
        zooms in within a rectangle around mouse coordinates while transforming 
        current cursor coordinates back to original coordinate space
        
        """
        if y <= 0:
            y = 1
        if x <= 0:
            x = 1

        x_prop, y_prop = x / self.canvas_width, y / self.canvas_height
        left_padding, right_padding = (
            int(round(x_prop * self.zoom.step_x)),
            int(round((1 - x_prop) * self.zoom.step_x)),
        )
        top_padding, bottom_padding = (
            int(round(y_prop * self.zoom.step_y)),
            int(round((1 - y_prop) * self.zoom.step_y)),
        )

        if self.zoom.flag > 0:
            x1, x2 = self.zoom.x1 + left_padding, self.zoom.x2 - right_padding
            y1, y2 = self.zoom.y1 + top_padding, self.zoom.y2 - bottom_padding
        if self.zoom.flag < 0:
            x1, x2 = (
                self.zoom.x1 - left_padding,
                self.zoom.x2 + right_padding,
            )
            y1, y2 = self.zoom.y1 - top_padding, self.zoom.y2 + bottom_padding
            if x1 < 0:
                x2 = x2 + abs(x1)
                x1 = 0
            if x2 > self.image_width:
                x1 = x1 - (x2 - self.image_width)
                x2 = self.image_width
            if y1 < 0:
                y2 = y2 + abs(y1)
                y1 = 0
            if y2 > self.image_height:
                y1 = y1 - (y2 - self.image_height)
                y2 = self.image_height

        ## failsafe when zooming out, sets zoom-coords to image coords
        if self.zoom.idx == 1:
            x1, x2, y1, y2 = 0, self.image_width, 0, self.image_height

        ## zoom coords
        self.zoom.x1, self.zoom.x2, self.zoom.y1, self.zoom.y2 = x1, x2, y1, y2

        ## global magnification factor
        self.zoom.global_fx = self.canvas_fx * (
            (self.zoom.x2 - self.zoom.x1) / self.image_width
        )
        self.zoom.global_fy = self.canvas_fy * (
            (self.zoom.y2 - self.zoom.y1) / self.image_height
        )

        ## update canvas
        self._canvas_mount(refresh=False)

        ## adjust brush size
        if self.tool == "draw":
            self.settings.line_width = int(
                self.line_width_orig
                / ((self.zoom.x2 - self.zoom.x1) / self.image_width)
            )
            
        ## redraw input
        if self.tool in ["comment", "labelling"]:
            y_pos, x_pos = self.settings.label_position
            self.canvas = copy.deepcopy(self.canvas_copy)
            cv2.putText(
                self.canvas,
                str(self.query) + ": " + str(self.data[_vars._comment_type]),
                (int(self.canvas.shape[0] * y_pos), int(self.canvas.shape[1] * x_pos)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.settings.label_size,
                self.settings.label_colour,
                self.settings.label_width,
                cv2.LINE_AA,
            )

    def _zoom_coords_orig(self, x, y):
        self.coords_original = (
            int(self.zoom.x1 + (x * self.zoom.global_fx)),
            int(self.zoom.y1 + (y * self.zoom.global_fy)),
        )


class _NoIndent(object):
    def __init__(self, value):
        # if not isinstance(value, (list, tuple, dict)):
        #     raise TypeError('Only lists and tuples can be wrapped')
        self.value = value
        
    def __repr__(self):
        return repr(self.value)
    
    def to_list(self):
        return self.value

class _NoIndentEncoder(json.JSONEncoder):

    FORMAT_SPEC = "@@{}@@"  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {"cls", "indent"}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(_NoIndentEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (
            self.FORMAT_SPEC.format(id(obj))
            if isinstance(obj, _NoIndent)
            else super(_NoIndentEncoder, self).default(obj)
        )

    def iterencode(self, obj, **kwargs):

        if isinstance(obj, np.intc):
            return int(obj)

        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(_NoIndentEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                    '"{}"'.format(format_spec.format(id)), json_repr
                )

            yield encoded


class _YamlFileMonitor:
    def __init__(self, filepath, delay=500):

        filepath = os.path.abspath(filepath)

        ## file, location and event action
        self.dirpath = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)
        self.filepath = filepath
        self.event_handler = PatternMatchingEventHandler(
            patterns=["*/" + self.filename]
        )
        self.event_handler.on_any_event = self._on_update

        ## intitialize
        self.content = _load_yaml(self.filepath)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.dirpath, recursive=False)
        self.observer.start()
        self.delay = delay
        self.time_start = None
        self.time_diff = 10

    def _on_update(self, event):

        if not self.time_start.__class__.__name__ == "NoneType":
            self.time_end = timer()
            self.time_diff = self.time_end - self.time_start

        if self.time_diff > 1:
            self.content = _load_yaml(self.filepath)
            config.window_close, config.pype_restart = True, True
            # cv2.destroyAllWindows()
            cv2.destroyWindow("phenopype")
            cv2.waitKey(self.delay)
        else:
            pass

        self.time_start = timer()
        
    # def _on_update(self, event):
    #     print("action")
    #     while cv2.getWindowProperty('phenopype', cv2.WND_PROP_VISIBLE) >= 0:
    #         print("waiting")
    #         cv2.destroyAllWindows()
    #         cv2.waitKey(self.delay)
    #         _config.window_close, _config.pype_restart = True, True


    def _stop(self):
        self.observer.stop()
        self.observer.join()

#%% functions - ANNOTATION helpers


def _get_annotation(
    annotations,
    annotation_type,
    annotation_id=None,
    reduce_counter=False,
    prep_msg=None,
    kwargs={},
):

    ## setup
    pype_mode = kwargs.get("pype_mode", False)
    prep_msg = kwargs.get("prep_msg", "")
    verbose = kwargs.get("verbose", False)

    annotations = copy.deepcopy(annotations)
    
    if annotations.__class__.__name__ == "NoneType":
        annotations = {}

    if not annotation_type.__class__.__name__ == "NoneType":
        annotation_id_str = annotation_type + "_id"
        print_msg = ""
    else:
        return {}

    ## get non-generic id for plotting
    if annotation_id_str in kwargs:
        annotation_id = kwargs.get(annotation_id_str)
        
    if annotations.__class__.__name__ in ["dict", "defaultdict"]:

        ## get ID from last used annotation function of that type
        if annotation_id.__class__.__name__ == "NoneType":

            if kwargs.get("annotation_counter"):
                print_msg = '- "{}" not provided: '.format(annotation_id_str)
                annotation_counter = kwargs.get("annotation_counter")
                annotation_id = string.ascii_lowercase[
                    annotation_counter[annotation_type]
                ]
                if annotation_id == "z":
                    print_msg = (
                        print_msg
                        + '- no precursing annotations of type "{}" found'.format(
                            annotation_type
                        )
                    )
                    annotation_id = None
                else:
                    if reduce_counter:
                        annotation_id = chr(ord(annotation_id) - 1)
                    print_msg = (
                        print_msg
                        + 'using last annotation of type "{}" with ID "{}"'.format(
                            annotation_type, annotation_id
                        )
                    )
            if annotation_type in annotations:
                    annotation_id = max(list(annotations[annotation_type].keys()))
                    print_msg = '"{}" not specified - using endmost in provided annotations: "{}"'.format(
                        annotation_id_str, annotation_id
                    )

            else:
                annotation = {}
                print_msg = '"{}" not specified and annotation type not found'.format(
                    annotation_id_str
                )

        ## check if type is given
        if annotation_type in annotations:

            ## extract item
            if annotation_id:
                if annotation_id in annotations[annotation_type]:
                    annotation = copy.deepcopy(annotations[annotation_type][annotation_id])
                else:
                    print_msg = 'could not find "{}" with ID "{}"'.format(
                        annotation_type, annotation_id
                    )
                    annotation = {}
            else:
                annotation = {}
        else:
            print_msg = 'incompatible annotation type supplied - need "{}" type'.format(
                annotation_type
            )
            annotation = {}


        ## cleaned feedback (skip identical messages)
        while True:
            if print_msg and verbose:
                if prep_msg:
                    print_msg = prep_msg + "\n\t" + print_msg
                if pype_mode:
                    if not print_msg == config.last_print_msg:
                        config.last_print_msg = print_msg
                        break
                    else:
                        pass
                else:
                    pass
                _print(print_msg)
                break
            break
    else:
        annotation = {}

    return annotation
    

def _get_annotation_id(
    annotations,
    annotation_type,
    annotation_id=None,
    reduce_counter=False,
    verbose=True,
    **kwargs,
):

    ## setup
    pype_mode = kwargs.get("pype_mode", False)
    annotation_counter = kwargs.get("annotation_counter", None)
    print_msg = None
    
    if annotation_id.__class__.__name__ == "NoneType":
        if annotation_counter:
            annotation_counter = kwargs.get("annotation_counter")
            annotation_id = string.ascii_lowercase[
                annotation_counter[annotation_type]
            ]
            if annotation_id == "z":
                print_msg = '- no precursing annotations of type "{}" found'.format(annotation_type)
                annotation_id = "a"
            else:
                if reduce_counter:
                    annotation_id = chr(ord(annotation_id) - 1)
                print_msg = '- using last created annotation of type \"{}\" with ID "{}"'.format(
                    annotation_type, annotation_id
                    )
        else:
            if annotation_type in annotations:
                annotation_id = max(list(annotations[annotation_type].keys()))
                print_msg = '- using endmost annotation of type "{}" with ID "{}"'.format(
                    annotation_type, annotation_id
                )

            else:
                annotation_id = "a"
                print_msg = '- annotation_id not specified and annotation of type "{}" not found'.format(
                    annotation_type
                )
    else:
        pass
    if verbose:
        _printer(print_msg, pype_mode)                

    return annotation_id

def _get_annotation_type(fun_name):

    annotation_type = _vars._annotation_functions[fun_name]    

    return annotation_type


def _get_annotation2(annotations, annotation_type, annotation_id, **kwargs):
    
    flag_failed = False
    
    if annotation_type in annotations:
        if annotation_id in annotations[annotation_type]:
            annotation = copy.deepcopy(annotations[annotation_type][annotation_id])     
        else:
            flag_failed = True
    else:
        flag_failed = True
        
    if flag_failed:   
        print('"get_annotation" failed!')
        annotation = {}
        
    return annotation
    

def _printer(print_msg, pype_mode=False,**kwargs):
    
    ## cleaned feedback (skip identical messages)
    while True:
        if print_msg and config.verbose:
            # if prep_msg:
            #     print_msg = prep_msg + "\n\t" + print_msg
            if pype_mode:
                if not print_msg == config.last_print_msg:
                    config.last_print_msg = print_msg
                    break
                else:
                    pass
            else:
                pass
            print(print_msg)
            break
        break



def _update_annotations(
    annotations, 
    annotation, 
    annotation_type, 
    annotation_id, 
    **kwargs,
):

    annotations = copy.deepcopy(annotations)
    
    if annotations.__class__.__name__ == "NoneType":
        annotations = {}
        
    if not annotation_type in annotations:
        annotations[annotation_type] = {}

    if annotation_id.__class__.__name__ == "NoneType":
        if "annotation_counter" in kwargs:
            annotation_counter = kwargs.get("annotation_counter")
            annotation_id = string.ascii_lowercase[annotation_counter[annotation_type]]
        else:
            annotation_id = "a"
            
    annotations[annotation_type][annotation_id] = copy.deepcopy(annotation)

    return annotations


#%% functions - GUI helpers

def _get_size(image_height, image_width, element="line_width", size_value="auto"):
    """
    Calculate automatic sizing for GUI elements based on image dimensions, or return the input size.
    
    Args:
        image (np.array): The image based on which sizing is calculated.
        element (str): The type of GUI element. Can be 'line', 'point', 'text', or 'text_width'.
        size_value (str or int): The size value or "auto" to calculate size dynamically.

    Returns:
        int: The calculated size for the specified GUI element, or the input size if not "auto".
    """
    # Check if the size_value is explicitly "auto"; if not, directly return the input if it's numeric
    if size_value != "auto":
        try:
            return int(size_value)  # Ensure it's a valid integer
        except ValueError:
            pass  # If it's not a valid integer, continue to calculate using default factors

    # Default factor dictionary
    default_factors = {
        "line_width": _vars.auto_line_width_factor,
        "node_size": _vars.auto_point_size_factor,
        "point_size": _vars.auto_point_size_factor,
        "label_size": _vars.auto_text_size_factor,
        "label_width": _vars.auto_text_width_factor,
        "text_size": _vars.auto_text_size_factor,
        "text_width": _vars.auto_text_width_factor,
    }

    # Retrieve factor from the default table
    factor = default_factors.get(element)
    
    # Calculate the diagonal of the image for scaling purposes
    image_diagonal = (image_height + image_width) / 2

    # Calculate and return the size based on the factor
    value = max(int(factor * image_diagonal), 1)

    return value


def _get_bgr(col_string, element=None):
    
    if col_string == "default" and element:
        default_color = getattr(_vars, f"_default_{element}")
        col_string = default_color
        
    if isinstance(col_string, str):
        col = Color(col_string)
        rgb = col.get_rgb()
        rgb_255 = [int(component * 255) for component in rgb]
        colour = tuple((rgb_255[2], rgb_255[1], rgb_255[0]))
        
    elif isinstance(col_string, int):
        colour = (col_string, col_string, col_string)
        
    return colour


def _get_GUI_data(annotation):

    data = []

    if annotation:
        if "info" in annotation:
            annotation_type = annotation["info"]["annotation_type"]
        if "data" in annotation:
            data = annotation["data"][annotation_type]

    return data


def _get_GUI_settings(kwargs, annotation=None):

    GUI_settings = {}

    if annotation:
        if "settings" in annotation:
            if "GUI" in annotation["settings"]:
                for key, value in annotation["settings"]["GUI"].items():
                    if not key in ["interactive"]:
                        GUI_settings[key] = value

    if kwargs:
        for key, value in kwargs.items():
            if key in _GUI_Settings.__annotations__:
                GUI_settings[key] = value
            elif key in ["interactive"]:
                pass

    return GUI_settings


#%% functions - YAML helpers


def _load_yaml(filepath, typ="rt", pure=False, legacy=False):

    ## this can read phenopype < 2.0 style config yaml files
    if legacy == True:

        def _construct_yaml_map(self, node):
            data = []
            yield data
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=True)
                val = self.construct_object(value_node, deep=True)
                data.append((key, val))

    else:

        def _construct_yaml_map(self, node):
            data = self.yaml_base_dict_type()
            yield data
            value = self.construct_mapping(node)
            data.update(value)

    SafeConstructor.add_constructor(u"tag:yaml.org,2002:map", _construct_yaml_map)
    yaml = YAML(typ=typ, pure=pure)
    yaml.indent(mapping=4, sequence=4, offset=4)

    if isinstance(filepath, (Path, str)):
        if Path(filepath).is_file():
            with open(filepath, "r") as file:
                return yaml.load(file)

        else:
            print("Cannot load file from specified filepath")
    else:
        print("Not a valid path - couldn't load yaml.")
        return


def _show_yaml(odict, ret=False, typ="rt"):

    yaml = YAML(typ=typ)
    yaml.indent(mapping=4, sequence=4, offset=4)

    if ret:
        with io.StringIO() as buf, redirect_stdout(buf):
            yaml.dump(odict, sys.stdout)
            return buf.getvalue()
    else:
        yaml.dump(odict, sys.stdout)
        
    
def _save_yaml(dictionary, filepath, typ="rt"):
    yaml = YAML(typ=typ)
    yaml.width = 160
    yaml.indent(mapping=4, sequence=4, offset=4)

    # Write YAML content to the temporary file
    temp_filepath = filepath + '.temp'
    with open(temp_filepath, 'w') as temp_file:
        yaml.dump(dictionary, temp_file)

    # Atomically replace the target file with the temporary file
    os.replace(temp_filepath, filepath)


def _yaml_flow_style(obj):
    if obj.__class__.__name__ == "dict":
        ret = ruamel.yaml.comments.CommentedMap(obj)
    elif obj.__class__.__name__ == "list":
        ret = ruamel.yaml.comments.CommentedSeq(obj) 
    ret.fa.set_flow_style()
    return ret


def _yaml_recursive_delete_comments(d):
    if isinstance(d, dict):
        for k, v in d.items():
            _yaml_recursive_delete_comments(k)
            _yaml_recursive_delete_comments(v)
    elif isinstance(d, list):
        for elem in d:
            _yaml_recursive_delete_comments(elem)
    try:
        # literal scalarstring might have comment associated with them
        attr = (
            "comment"
            if isinstance(d, ruamel.yaml.scalarstring.ScalarString)
            else ruamel.yaml.comments.Comment.attrib
        )
        delattr(d, attr)
    except AttributeError:
        pass


#%% functions - DIALOGS

def _overwrite_check_file(path, overwrite):
    
    filename = os.path.basename(path)
    
    if os.path.isfile(path) and overwrite == False:
        print(
            filename + " not saved - file already exists (overwrite=False)."
        )
        return False
    elif os.path.isfile(path) and overwrite == True:
        print(filename + " saved under " + path + " (overwritten).")
        return True
    elif not os.path.isfile(path):
        print(filename + " saved under " + path + ".")
        return True
    
def _overwrite_check_dir(path, overwrite):
    
    dirname = os.path.basename(path)
    
    if os.path.isdir(path) and overwrite == False:
        print(
            dirname + " not saved - file already exists (overwrite=False)."
        )
        return False
    elif os.path.isdir(path) and overwrite == True:
        print(dirname + " saved under " + path + " (overwritten).")
        return True
    elif not os.path.isdir(path):
        print(dirname + " saved under " + path + ".")
        return True
    
#%% functions - PRINTING / LOGGING

def _print(msg, lvl=0, **kwargs):
    if config.verbose:
        if lvl >= config.verbosity_level:
            print(msg)
        

# def _label_formatter(label_dict):
#     old_dict = copy.deepcopy(label_dict)
#     new_dict = {}
#     for key, val in old_dict.items():
#         if key == "mask":
#             if "coords" in val:
#                 val["coords"] = np.array(val["coords"])
#         new_dict[key] = val
#     return new_dict
    
#%% functions - CONTOURS

def _get_orientation(coords, method="ellipse"):
    
    if method=="ellipse":
        ellipse = cv2.fitEllipse(coords)
        center,diameter,angle = ellipse
        return angle
    
    if method=="pca":
        
        # Construct a buffer used by the pca analysis
        sz = len(coords)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
          data_pts[i,0] = coords[i,0,0]
          data_pts[i,1] = coords[i,0,1]
           
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        angle = np.rad2deg(angle)
        
    return angle
    
def _resize_contour(contour, img_orig, img_resized):

    coef_y = img_orig.shape[0] / img_resized.shape[0]
    coef_x = img_orig.shape[1] / img_resized.shape[1]
    
    contour[:, :, 0] = contour[:, :, 0] * coef_x
    contour[:, :, 1] = contour[:, :,  1] * coef_y

    return contour

# def _rotate_coords(array, center, angle, offset=(0,0)):
        
#     dt = array.dtype
    
#     radian = np.deg2rad(angle)
#     rotation_matrix = np.array([[np.cos(radian),np.sin(radian)],[-np.sin(radian),np.cos(radian)]])
#     rotated_array = np.dot(array - center, rotation_matrix) + _rotate_point(center, angle) + offset
    
#     return rotated_array.astype(dt)

def _rotate_coords(coords, angle):
    
    cnt = copy.deepcopy(coords)
    
    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho


    def pol2cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y
    
    angle = angle * -1
    
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def _rotate_point(point, angle, center=None):
    
    x, y = point
    radians = angle * (pi/180)
    
    if center:
        center_x, center_y = center
        x_rotated = round(((x - center_x) * cos(radians)) - ((y - center_y) * sin(radians)) + center_x) 
        y_rotated = round(((x - center_x) * sin(radians)) + ((y - center_y) * cos(radians)) + center_y)

    else:
        x_rotated = abs(round(cos(radians) * x + sin(radians) * y))
        y_rotated = abs(round(-sin(radians) * x + cos(radians) * y))
        
    return x_rotated, y_rotated


def _extract_roi_center(image, coords, dim_final):
    
    ## get half of final length
    dim_half = int(dim_final/2)
    
    # Calculate the center of the contour
    M = cv2.moments(coords)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # Default to the center of the image if contour area is 0
        cY, cX = image.shape[0] // 2, image.shape[1] // 2
    
    # Define the top-left corner of the 512x512 area
    start_x = max(cX - dim_half, 0)
    start_y = max(cY - dim_half, 0)
    
    # Adjust the bottom right corner if it goes beyond the image dimensions
    end_x = min(start_x + dim_final, image.shape[1])
    end_y = min(start_y + dim_final, image.shape[0])
    
    # Adjust the start points accordingly if the end points were adjusted
    if end_x - start_x < dim_final:
        start_x = max(end_x - dim_final, 0)
    if end_y - start_y < dim_final:
        start_y = max(end_y - dim_final, 0)
    
    return image[start_y:end_y, start_x:end_x], (start_y, end_y,start_x,end_x)


def _calc_contour_stats(contour, mode="circle"):
    if mode=="moments":
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = [cx, cy]
        # Calculate maximum distance from centroid to contour points for an approximated diameter
        max_dist = max(np.linalg.norm(np.array([px, py]) - np.array(center)) for px, py in contour[:, 0, :])
        diameter = int(max_dist * 2)  # Diameter is twice the maximum distance
        area = int(cv2.contourArea(contour))
    elif mode=="circle":
        center, radius = cv2.minEnclosingCircle(contour)
        center = [int(center[0]), int(center[1])]
        diameter = int(radius * 2)
        area = int(cv2.contourArea(contour))
    elif mode=="rectangle":
        x, y, w, h = cv2.boundingRect(contour)
        center = [x + w // 2, y + h // 2]  # Center of the bounding box
        diameter = int((w**2 + h**2)**0.5)  # Approximate diameter from the diagonal of the bounding box
        area = int(cv2.contourArea(contour))  # Area remains calculated by contourArea for accuracy
    
    return center, area, diameter


#%% functions - VARIOUS

def _calc_distance_2point(x1,x2,y1,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)
    
def _calc_distance_polyline(coords):
    distances = []
    for i in range(len(coords)-1):
        current_line = coords[i]
        next_line = coords[i+1]
        distances.append(_calc_distance_2point(current_line[0],next_line[0],current_line[1],next_line[1]))
    return sum(distances)

def _convert_box_xywh_to_xyxy(box):
    if len(box) == 4:
        return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    else:
        result = []
        for b in box:
            b = _convert_box_xywh_to_xyxy(b)
            result.append(b)               
    return result


def _convert_arr_tup_list(arr_list, add_first=False):

    if not arr_list.__class__.__name__ == "list":
        arr_list = [arr_list]

    tup_list = []
    for array in arr_list:
        point_list = []
                
        if type(array) == np.ndarray and len(array.shape) == 3 and not array.shape[1] == 1:
            array = array[0]
                 
        for idx, point in enumerate(array):
            if type(array) == np.ndarray and point.shape[0] == 1:
                point_converted = tuple((int(point[0][0]), int(point[0][1])))
            elif type(array) == list or point.shape[0] > 1:
                point_converted = tuple((int(point[0]), int(point[1])))
            point_list.append(point_converted)
            if idx == 0:
                first_point = point_converted
                
        ## add first point during contour->mask conversion
        if add_first:
            point_list.append(first_point)
            
        tup_list.append(point_list)
        
    return tup_list


def _convert_tup_list_arr(tup_list):
    
    array_list = []
    if type(tup_list[0]) == list and len(tup_list[0])>2:
        for points in tup_list:
            point_list = []
            for point in points:
                point_list.append([list(point)])
            array_list.append(np.asarray(point_list, dtype=np.int32))
    elif type(tup_list[0]) == list or type(tup_list[0]) == tuple:
        array_list = np.array([np.array([elem], dtype=np.int32) for elem in tup_list], dtype=np.int32)
        
    return array_list


def _check_pype_tag(tag):

    if tag.__class__.__name__ == "str":

        ## pype name check
        if "pype_config" in tag:
            tag = tag.replace("pype_config", "")
            print('Do not add "pype_config", only a short tag')
        if ".yaml" in tag:
            tag = tag.replace(".yaml", "")
            print("Do not add extension, only a short tag")
        if "_" in tag:
            raise SyntaxError("Underscore not allowed in pype tag - aborting.")
        for char in "[@!#$%^&*()<>?/|}{~:]\\":
            if char in tag:
                raise SyntaxError(
                    "No special characters allowed in pype tag - aborting."
                )


def _create_mask_bin(image, contours):
    mask_bin = np.zeros(image.shape[0:2], np.uint8)
    if (
        contours[0].__class__.__name__ == "list"
        or contours.__class__.__name__ == "list"
    ):
        cv2.fillPoly(mask_bin, [np.array(contours, dtype=np.int32)], _get_bgr("white"))
    elif contours[0].__class__.__name__ == "ndarray":
        for contour in contours:
            cv2.fillPoly(
                mask_bin, [np.array(contour, dtype=np.int32)], _get_bgr("white")
            )
    return mask_bin


def _create_mask_bool(image, contours):
    mask_bin = _create_mask_bin(image, contours)
    return np.array(mask_bin, dtype=bool)


def _decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def _del_rw(action, name, exc):
    os.chmod(name, S_IWRITE)
    os.remove(name)


def _equalize_histogram(image, detected_rect_mask, template):
    """Histogram equalization via interpolation, upscales the results from the detected reference card to the entire image.
    May become a standalone function at some point in the future. THIS STRONGLY DEPENDS ON THE QUALITY OF YOUR TEMPLATE.
    Mostly inspired by this SO question: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    More theory here: https://docs.opencv.org/master/d4/d1b/tutorial_histogram_equalization.html
    """
    detected_ravel = detected_rect_mask.ravel()
    template_ravel = template.ravel()

    detected_counts = np.bincount(detected_ravel, minlength=256)
    detected_quantiles = np.cumsum(detected_counts).astype(np.float64)
    detected_quantiles /= detected_quantiles[-1]

    template_values = np.arange(0, 256, 1, dtype=np.uint8)
    template_counts = np.bincount(template_ravel, minlength=256)
    template_quantiles = np.cumsum(template_counts).astype(np.float64)
    template_quantiles /= template_quantiles[-1]

    interp_template_values = np.interp(
        detected_quantiles, template_quantiles, template_values
    )
    interp_template_values = interp_template_values.astype(image.dtype)

    return interp_template_values[image]


def _file_walker(
    directory,
    filetypes=[],
    include=[],
    include_all=True,
    exclude=[],
    recursive=False,
    unique="path",
    **kwargs
):
    """
    
    Parameters
    ----------
    directory : str
        path to directory to search for files
    recursive: (optional): bool,
        "False" searches only current directory for valid files; "True" walks 
        through all subdirectories
    filetypes (optional): list of str
        single or multiple string patterns to target files with certain endings
    include (optional): list of str
        single or multiple string patterns to target certain files to include
    include_all (optional): bool,
        either all (True) or any (False) of the provided keywords have to match
    exclude (optional): list of str
        single or multiple string patterns to target certain files to exclude - can overrule "include"
    unique (optional): str (default: "filepath")
        how should unique files be identified: "filepath" or "filename". "filepath" is useful, for example, 
        if identically named files exist in different subfolders (folder structure will be collapsed and goes into the filename),
        whereas filename will ignore all those files after their first occurrence.

    Returns
    -------
    None.

    """
    ## kwargs
    pype_mode = kwargs.get("pype_mode", False)
    if not filetypes.__class__.__name__ == "list":
        filetypes = [filetypes]
    if not include.__class__.__name__ == "list":
        include = [include]
    if not exclude.__class__.__name__ == "list":
        exclude = [exclude]
    flag_include_all = include_all
    flag_recursive = recursive
    flag_unique = unique

    ## find files
    filepaths1, filepaths2, filepaths3, filepaths4 = [], [], [], []
    if flag_recursive == True:
        for root, dirs, files in os.walk(directory):
            for file in os.listdir(root):
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):
                    filepaths1.append(filepath)
    else:
        for file in os.listdir(directory):
            filepath = os.path.join(directory, file)
            if os.path.isfile(filepath):
                filepaths1.append(filepath)

    ## file endings
    if len(filetypes) > 0:
        for filepath in filepaths1:
            if filepath.endswith(tuple(filetypes)):
                filepaths2.append(filepath)
    elif len(filetypes) == 0:
        filepaths2 = filepaths1

    ## include
    if len(include) > 0:
        for filepath in filepaths2:
            if flag_include_all:
                if all(inc in os.path.basename(filepath) for inc in include):
                    filepaths3.append(filepath)
            else:
                if pype_mode:
                    if any(inc in Path(filepath).stem for inc in include):
                        filepaths3.append(filepath)
                else:
                    if any(inc in os.path.basename(filepath) for inc in include):
                        filepaths3.append(filepath)
    else:
        filepaths3 = filepaths2

    ## exclude
    if len(exclude) > 0:
        for filepath in filepaths3:
            if not any(exc in os.path.basename(filepath) for exc in exclude):
                filepaths4.append(filepath)
    else:
        filepaths4 = filepaths3

    ## check if files found
    filepaths = filepaths4
    if len(filepaths) == 0 and not pype_mode:
        print("No files found under the given location that match given criteria.")
        return [], []
    
    ## allow unique filenames filepath or by filename only
    filenames, unique_filename, unique, duplicate = [], [], [], []
    for filepath in filepaths:
        filenames.append(os.path.basename(filepath))
    if flag_unique in ["filepaths", "filepath", "path"]:
        for filename, filepath in zip(filenames, filepaths):
            if not filepath in unique:
                unique.append(filepath)
            else:
                duplicate.append(filepath)
    elif flag_unique in ["filenames", "filename", "name"]:
        for filename, filepath in zip(filenames, filepaths):
            if not filename in unique_filename:
                unique_filename.append(filename)
                unique.append(filepath)
            else:
                duplicate.append(filepath)

    return unique, duplicate


def _get_caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

       An empty string is returned if skipped levels exceed stack height
    """
    
    ## inspect stack
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
      return ''
    parentframe = stack[start][0]    
    
    ## get parentframe
    codename = parentframe.f_code.co_name
    
    # detect classname
    if 'self' in parentframe.f_locals and codename== "__init__":
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name = parentframe.f_locals['self'].__class__.__name__
    
    else:
        if codename != '<module>':  # top level usually
            name = codename  # function or a method
        else:
            module = inspect.getmodule(parentframe)
            if module:
                name = module.__name__
            # `modname` can be None when frame is executed directly in console
            # TODO(techtonik): consider using __main__
            # module = inspect.getmodule(parentframe)
            # if module:
            #     name = module.__name__
    
    ## Avoid circular refs and frame leaks
    #  https://docs.python.org/2.7/library/inspect.html#the-interpreter-stack
    del parentframe, stack

    return name


def _load_project_image_directory(dir_path, tag=None, as_container=True, **kwargs):
    """
    Parameters
    ----------
    dirpath: str
        path to a phenopype project directory containing raw image, attributes 
        file, masks files, results df, etc.
    tag : str
        pype suffix that is appended to all output
        
    Returns
    -------
    container
        A phenopype container is a Python class where loaded images, 
        dataframes, detected contours, intermediate output, etc. are stored 
        so that they are available for inspection or storage at the end of 
        the analysis. 

    """

    ## check if directory
    if not os.path.isdir(dir_path):
        print("Not a valid phenoype directory - cannot load files.")
        return

    ## check if attributes file and load otherwise
    if not os.path.isfile(os.path.join(dir_path, "attributes.yaml")):
        print("Attributes file missing - cannot load files.")
        return
    else:
        attributes = _load_yaml(os.path.join(dir_path, "attributes.yaml"))

    ## check if requires info is contained in attributes and load image
    if not "image_phenopype" in attributes or not "image_original" in attributes:
        print("Attributes doesn't contain required meta-data - cannot load files.")
        return

    ## load image
    if attributes["image_phenopype"]["mode"] == "link":
        file_path = attributes["image_phenopype"]["filepath"]
        image_path = os.path.join(
            dir_path, file_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Link mode: did not find image - images folder set up correctly?")
    else:
        image_path = os.path.join(dir_path, attributes["image_phenopype"]["filename"])
    image = utils.load_image(image_path)

    ## return
    if as_container:
        return _Container(
            image=image, 
            dir_path=dir_path, 
            file_suffix=tag, 
            tag=tag,
            image_name = attributes["image_original"]["filename"],
            )
    else:
        return image


def _load_image_data(image_path, path_and_type=True, image_rel_path=None, resize=1):
    """
    Create a DataFreame with image information (e.g. dimensions).

    Parameters
    ----------
    image: str or ndarray
        can be a path to an image stored on the harddrive OR an array already 
        loaded to Python.
    path_and_type: bool, optional
        return image path and filetype to image_data dictionary

    Returns
    -------
    image_data: dict
        contains image data (+meta data, if selected)

    """
    
    if image_path.__class__.__name__ == "str":
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            width, height = image.size
            image.close()
            image_data = {
                "filename": os.path.split(image_path)[1],
                "width": width,
                "height": height,
            }

            if path_and_type:
                if not image_rel_path.__class__.__name__ == "NoneType":
                    image_path = image_rel_path
                
                image_data.update(
                    {
                        "filepath": image_path,
                        "filetype": os.path.splitext(image_path)[1],
                    }
                )
        else:
            raise FileNotFoundError("Invalid image path - could not load image.")
    else:
        raise TypeError("Not a valid image file - cannot read image data.")


    ## issue warnings for large images
    if width * height > 125000000:
        warnings.warn("Large image - expect slow processing.")
    elif width * height > 250000000:
        warnings.warn(
            "Extremely large image - expect very slow processing \
                      and consider resizing."
        )

    ## return image data
    return image_data



def _print_mod(msg, context="caller", level=1):
    if context=="caller":
        caller = _get_caller_name(level)
        print(caller + ":", msg)
    elif context=="none":
        print(msg)
        
def _pprint_fill_hbar(message, symbol="-", ret=False):
    terminal_width = shutil.get_terminal_size()[0]
    message_length = len(message)

    if message_length >= terminal_width:
        formatted_message = message
    else:
        bar_length = (terminal_width - message_length - 2) // 2
        horizontal_bar = symbol * bar_length
        formatted_message = f"{horizontal_bar} {message} {horizontal_bar}"
        residual = terminal_width - len(formatted_message)
        formatted_message = formatted_message + symbol * residual
        
    if not ret:
        print(formatted_message)
    else:
        return formatted_message

def _pprint_hbar(symbol="-", ret=False):
    terminal_width = os.get_terminal_size()[0]
    string = symbol * terminal_width
    if not ret:
        print(string)
    else:
        return string


def _resize_mask(original_bbox, resize_x, resize_y):

    resized_bbox = (
        int(original_bbox[0] * resize_x),
        int(original_bbox[1] * resize_y),
        int(original_bbox[2] * resize_x),
        int(original_bbox[3] * resize_y)
    )
    
    return resized_bbox


def _rotate_image(image, angle, ret_center=False):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    Source: https://stackoverflow.com/a/47248339/5238559
    """
    
    height, width = image.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    # rotate image with the new bounds and translated rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    
    if ret_center:
        return rotated_image, image_center
    else:
        return rotated_image


def _save_prompt(object_type, filepath, ow_flag):

    if os.path.isfile(filepath) and ow_flag == False:
        print_msg = "- {} not saved - file already exists (overwrite=False)".format(
            object_type
        )
        ret = False
    elif os.path.isfile(filepath) and ow_flag == True:
        print_msg = "- {} saved under {} (overwritten)".format(object_type, filepath)
        ret = True
    elif not os.path.isfile(filepath):
        print_msg = "- {} saved under {}".format(object_type, filepath)
        ret = True

    print(print_msg)
    return ret
