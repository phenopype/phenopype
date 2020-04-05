#%% modules

import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from pprint import PrettyPrinter
from PIL import Image, ExifTags
from ruamel.yaml import YAML

from phenopype.utils_lowlevel import _image_viewer 
from phenopype.utils_lowlevel import _load_yaml, _show_yaml, _save_yaml, _yaml_file_monitor
from phenopype.settings import *
from phenopype.core.export import *
from phenopype.core.visualization import *

#%% settings

Image.MAX_IMAGE_PIXELS = 999999999
pretty = PrettyPrinter(width=30)

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return "WARNING: " + str(msg) + '\n'
warnings.formatwarning = custom_formatwarning
warnings.simplefilter('always', UserWarning)

#%% classes

class container(object):
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
        or load_directory).
    df_image_data: DataFrame
        a dataframe that contains meta-data of the provided image to be passed on
        to all results-DataFrames
    save_suffix : str, optional
        suffix to append to filename of results files

    """
    def __init__(self, image, df_image_data, save_suffix=None):

        ## images
        self.image = image
        self.image_copy = copy.deepcopy(self.image)
        self.image_mod = copy.deepcopy(self.image)
        self.image_bin = None
        self.image_gray = None
        self.canvas = copy.deepcopy(self.image)
        
        ## data frames
        self.df_image_data = df_image_data
        self.df_image_data_copy = copy.deepcopy(self.df_image_data)

        ## attributes
        self.dirpath = None
        self.save_suffix = None
        self.scale_px_mm_ratio = None

    def load(self, contours=False, **kwargs):
        """
        Autoload function for container: loads results files with given save_suffix
        into the container. Can be used manually, but is typically used within the
        pype routine.
        
        Parameters
        ----------
        save_suffix : str, optional
            suffix to include when looking for files to load

        """
        files, loaded = [], []

        ## data flags
        flag_contours = contours


        ## data from attributes file
        attr_path = os.path.join(self.dirpath, "attributes.yaml")
        if os.path.isfile(attr_path):

            ## other data
            if not hasattr(self, "df_other_data"):
                attr = _load_yaml(attr_path)
                if "other" in attr:
                    self.df_other_data = pd.DataFrame(attr["other"], index=[0])
                    loaded.append("columns " + ', '.join(list(self.df_other_data)) + " from attributes.yaml")
    
            ## scale
            if not hasattr(self, "scale_template_px_mm_ratio"):
                attr = _load_yaml(attr_path)
                if "scale" in attr:
                    self.scale_template_px_mm_ratio = attr["scale"]["template_px_mm_ratio"]
                    if "current_px_mm_ratio" in attr["scale"]:
                        self.scale_current_px_mm_ratio = attr["scale"]["current_px_mm_ratio"]
                        loaded.append("scale information loaded from attributes.yaml")
                    if "template_path" in attr["scale"]:
                        self.scale_template = cv2.imread(attr["scale"]["template_path"])
                        loaded.append("template loaded from root directory")

            ## scale
            if not hasattr(self, "df_draw"):
                attr = _load_yaml(attr_path)
                if "drawing" in attr:
                    self.df_draw = pd.DataFrame(attr["drawing"], index=[0])
                    loaded.append("drawing loaded from attributes.yaml")

        if self.save_suffix.__class__.__name__ == 'NoneType':
            if "save_suffix" in kwargs:
                self.save_suffix = kwargs.get("save_suffix")
            else:
                print("Save_suffix missing - not loading saved results.")
                return

        if self.save_suffix:
            for file in os.listdir(self.dirpath):
                if self.save_suffix in file and not "pype_config" in file:
                    files.append(file[0:file.rindex('_')])
        else:
            for file in os.listdir(self.dirpath):
                files.append(file[0:file.rindex('.')])

        ## contours
        if flag_contours:
            if not hasattr(self, "df_contours") and "contours" in files:
                path = os.path.join(self.dirpath, "contours_" + self.save_suffix + ".csv")
                if os.path.isfile(path):
                    self.df_contours = pd.read_csv(path) 
                    loaded.append("contours_" + self.save_suffix + ".csv")

        ## landmarks
        if not hasattr(self, "df_landmarks") and "landmarks" in files:
            path = os.path.join(self.dirpath, "landmarks_" + self.save_suffix + ".csv")
            if os.path.isfile(path):
                self.df_landmarks = pd.read_csv(path) 
                loaded.append("landmarks_" + self.save_suffix + ".csv")

        ## polylines
        if not hasattr(self, "df_polylines") and "polylines" in files:
            path = os.path.join(self.dirpath, "polylines_" + self.save_suffix + ".csv")
            if os.path.isfile(path):
                self.df_polylines = pd.read_csv(path) 
                loaded.append("polylines_" + self.save_suffix + ".csv")

        ## masks
        if not hasattr(self, "df_masks") and "masks" in files:
            path = os.path.join(self.dirpath, "masks_" + self.save_suffix + ".csv")
            if os.path.isfile(path):
                self.df_masks = pd.read_csv(path) 
                loaded.append("masks_" + self.save_suffix + ".csv")

        ## feedback
        if len(loaded)>0:
            print("AUTOLOAD\n- " + '\n- '.join(loaded) )



    def reset(self):
        """
        Resets modified images, canvas and df_image_data to original state. Can be used manually, but is typically used within the
        pype routine.

        """
        self.image = copy.deepcopy(self.image_copy)
        self.canvas = copy.deepcopy(self.image_copy)
        self.df_image_data = copy.deepcopy(self.df_image_data_copy)

        # if hasattr(self, "df_masks"):
        #     del(self.df_masks)

        # if hasattr(self, "df_contours"):
        #     del(self.df_contours)

    def save(self, export_list=[], overwrite=False):
        """
        Autosave function for container. 

        Parameters
        ----------
        export_list: list, optional
            used in pype rountine to check against already performed saving operations.
            running container.save() with an empty export_list will assumed that nothing
            has been saved so far, and will try 
        overwrite : bool, optional
            gloabl overwrite flag in case file exists

        """

        ## kwargs 
        flag_overwrite = overwrite

        ## feedback
        print("AUTOSAVE")

        ## visualization

        ## canvas
        if not self.canvas.__class__.__name__ == "NoneType" and not "save_canvas" in export_list:
            print("save_canvas")
            save_canvas(self)

        ## results

        ## colours
        if hasattr(self, "df_colours") and not "save_colours" in export_list:
            print("save_colours")
            save_colours(self, overwrite=flag_overwrite)

        ## contours
        if hasattr(self, "df_contours") and not "save_contours" in export_list:
            print("save_contours")
            save_contours(self, overwrite=flag_overwrite)

        ## entered data
        if hasattr(self, "df_other_data") and not "save_data_entry" in export_list:
            print("save_data_entry")
            save_data_entry(self, overwrite=flag_overwrite)

        ## landmarks
        if hasattr(self, "df_landmarks") and not "save_landmarks" in export_list:
            print("save_landmarks")
            save_landmarks(self, overwrite=flag_overwrite)

        ## masks
        if hasattr(self, "df_masks") and not "save_masks" in export_list:
            print("save_masks")
            save_masks(self, overwrite=flag_overwrite)

        ## polylines
        if hasattr(self, "df_polylines") and not "save_polylines" in export_list:
            print("save_polylines")
            save_polylines(self, overwrite=flag_overwrite)

        ## other data
        
        ## drawing
        if hasattr(self, "df_draw") and not "save_drawing" in export_list:
            print("save_drawing")
            save_drawing(self, overwrite=True)

        ## scale
        if hasattr(self, "scale_current_px_mm_ratio") and not "save_scale" in export_list:
            print("save_scale")
            save_scale(self, overwrite=True)

    # def show(self, **kwargs):
    #     """
        
    
    #     Parameters
    #     ----------
    #     components : TYPE, optional
    #         DESCRIPTION. The default is [].
    #     **kwargs : TYPE
    #         DESCRIPTION.
    
    #     Returns
    #     -------
    #     None.
    
    #     cfg = "pype_config_v1.yaml"
    #     cfg[cfg.rindex('_')+1:cfg.rindex('.')]
    
    #     """
    #     ## kwargs
    #     show_list = kwargs.get("show_list",[])

    #     ## feedback
    #     print("AUTOSHOW")
        
        # ## contours
        # if hasattr(self, "df_contours") and not "show_contours" in show_list:
        #     print("show_contours")
        #     show_contours(self)
    
        # ## landmarks
        # if hasattr(self, "df_landmarks") and not "show_landmarks" in show_list:
        #     print("show_landmarks")
        #     show_landmarks(self)
    
        # ## masks
        # if hasattr(self, "df_masks") and not "show_masks" in show_list:
        #     print("show_masks")
        #     show_masks(self)
    
        # ## polylines
        # if hasattr(self, "df_polylines") and not "show_polylines" in show_list:
        #     print("show_polylines")
        #     show_polylines(self)

#%% functions
            
def load_directory(directory_path, cont=True, df=True, meta=True, resize=1, 
                   **kwargs):
    """
    Parameters
    ----------
    directory_path: str or ndarray
        path to a phenopype project directory containing raw image, attributes 
        file, masks files, results df, etc.
    cont: bool, optional
        should the loaded image (and DataFrame) be returned as a phenopype 
        container
    df: bool, optional
        should a DataFrame containing image information (e.g. dimensions) 
        be returned.
    meta: bool, optional
        should the DataFrame encompass image meta data (e.g. from exif-data). 
        This works only when obj_input is a path string to the original file.
    resize: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of
        original size).
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
    ## kwargs
    ## kwargs 
    flag_resize = resize
    flag_df = df
    flag_meta = meta
    flag_container = cont
    exif_fields = kwargs.get("fields", default_meta_data_fields)
    if not exif_fields.__class__.__name__ == "list":
        exif_fields = [exif_fields]

    ## check if directory
    if not os.path.isdir(directory_path):
        sys.exit("Not a valid phenoype directory - cannot load files.")

    ## load attributes-file
    attr = _load_yaml(os.path.join(directory_path, "attributes.yaml"))
    image = cv2.imread(attr["project"]["raw_path"])
    df_image_data = pd.DataFrame({"filename": attr["image"]["filename"],
                       "width": attr["image"]["width"],
                       "height": attr["image"]["height"]
                       }, index=[0])
    
    if "size_ratio_original "in attr["image"]:
        df_image_data["size_ratio_original"] = attr["image"]["size_ratio_original"]

    # ## add meta-data 
    # if flag_meta:
    #     exif_data_all, exif_data = attr["meta"], {}
    #     for field in exif_fields:
    #         if field in exif_data_all:
    #             exif_data[field] = exif_data_all[field]
    #     exif_data = dict(sorted(exif_data.items()))
    #     df_image_data = pd.concat([df_image_data.reset_index(drop=True), 
    # pd.DataFrame(exif_data, index=[0])], axis=1)

    ## return
    if flag_container == True:
        ct = container(image, df_image_data)
        ct.dirpath = directory_path
        ct.image_data = attr["image"]

    # ## other, saved data to pass on
    # if "other" in attr:
    #    ct.df_other = pd.DataFrame(attr["other"], index=[0])

    if flag_container == True:
        return ct
    elif flag_container == False:
        if flag_df:
            return image, df
        else:
            return image



def load_image(obj_input, cont=False, df=False, meta=False, resize=1, **kwargs):
    """
    Create ndarray from image path or return or resize exising array.

    Parameters
    ----------
    obj_input: str or ndarray
        can be a path to an image stored on the harddrive OR an array already 
        loaded to Python.
    cont: bool, optional
        should the loaded image (and DataFrame) be returned as a phenopype 
        container
    df: bool, optional
        should a DataFrame containing image information (e.g. dimensions) be 
        returned.
    meta: bool, optional
        should the DataFrame encompass image meta data (e.g. from exif-data). 
        This works only when obj_input is a path string to the original file.
    resize: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of 
        original size).
    kwargs: 
        developer options

    Returns
    -------
    ct: container
        A phenopype container is a Python class where loaded images, 
        dataframes, detected contours, intermediate output, etc. are stored 
        so that they are available for inspection or storage at the end of 
        the analysis. 
    image: ndarray
        original image (resized, if selected)
    df_image_data: DataFrame
        contains image data (+meta data), if selected

    """
    ## kwargs 
    flag_resize = resize
    flag_df = df
    flag_meta = meta
    flag_container = cont
    exif_fields = kwargs.get("fields", default_meta_data_fields)
    if not exif_fields.__class__.__name__ == "list":
        exif_fields = [exif_fields]

    ## method
    if obj_input.__class__.__name__ == "str":
        if os.path.isfile(obj_input):
            image = cv2.imread(obj_input)
            dirpath = os.path.split(obj_input)[0]
        else:
            sys.exit("Invalid image path - cannot load image from str.")
    elif obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        dirpath = os.getcwd()
    else:
        sys.exit("Invalid input format - cannot load image.")

    ## resize
    if flag_resize < 1:
        image = cv2.resize(image, (0,0), fx=1*flag_resize, fy=1*flag_resize, 
                           interpolation=cv2.INTER_AREA)

    ## load image data
    image_data = load_image_data(obj_input, flag_resize)
    df_image_data = pd.DataFrame({"filename": image_data["filename"],
          "width": int(image_data["width"]*flag_resize),
          "height": int(image_data["height"]*flag_resize),
          "size_ratio_original": flag_resize}, index=[0])

    ## add meta-data 
    if flag_meta:
        meta_data = load_meta_data(obj_input, fields=exif_fields)        
        df_image_data = pd.concat([df_image_data.reset_index(drop=True), 
                                   pd.DataFrame(meta_data, index=[0])], 
                                  axis=1)

    ## return
    if flag_container == True:
        ct = container(image, df_image_data)
        ct.image_data = image_data
        ct.dirpath = dirpath
        return ct
    elif flag_container == False:
        if flag_df or flag_meta:
            return image, df_image_data
        else:
            return image



def load_image_data(obj_input, resize=1):
    """
    Create a DataFreame with image information (e.g. dimensions).

    Parameters
    ----------
    obj_input: str or ndarray
        can be a path to an image stored on the harddrive OR an array already 
        loaded to Python.
    resize: float
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of 
        original size). gets stored to a DataFrame column

    Returns
    -------
    image_data: dict
        contains image data (+meta data, if selected)

    """
    if obj_input.__class__.__name__ == "str":
        if os.path.isfile(obj_input):
            path = obj_input
        elif os.path.isdir(obj_input):
            attr = _load_yaml(os.path.join(obj_input, "attributes.yaml"))
            path = attr["project"]["raw_path"]
        image = Image.open(path)
        width, height = image.size
        image.close()
        image_data = {
            "filename": os.path.split(obj_input)[1],
            "filepath": obj_input,
            "filetype": os.path.splitext(obj_input)[1],
            "width": int(width*resize),
            "height": int(height*resize),
            "size_ratio_original": resize
            }
    elif obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        width, height = image.shape[0:2]
        image_data = {
                "filename": "unknown",
                "filepath": "unkown",
                "filetype": "array_type",
                "width": int(width*resize),
                "height": int(height*resize),
                "size_ratio_original": resize
            }
    else:
        warnings.warn("Not a valid image file - cannot read image data.")

    ## issue warnings for large images
    if width*height > 125000000:
        warnings.warn("Large image - expect slow processing and consider \
                      resizing.")
    elif width*height > 250000000:
        warnings.warn("Extremely large image - expect very slow processing \
                      and consider resizing.")

    ## return image data
    return image_data



def load_meta_data(image_path, show_fields=False, 
                   fields=default_meta_data_fields):
    """
    Extracts metadata (mostly Exif) from original image

    Parameters
    ----------
    image_path : str 
        path to image on harddrive
    show_fields: bool, optional
        show which exif data fields are available from given image
    fields: list or str
        which exif data fields should be extracted. default fields can be 
        modified in settings

    Returns
    -------
    exif_data: dict
        image meta data (exif)

    """
    ## kwargs
    flag_show = show_fields
    exif_fields = fields
    if not exif_fields.__class__.__name__ == "list":
        exif_fields = [exif_fields]

    # ## check if basic fields are present
    # if exif_fields.__class__.__name__ == "list":
    #     if not len(exif_fields) == 0:
    #         prepend = list(set(default_fields) - set(exif_fields))
    #         exif_fields = prepend + exif_fields

    ## read image
    if image_path.__class__.__name__ == "str":
        if os.path.isfile(image_path):
            image = Image.open(image_path)
        else:
            print("Not a valid image file - cannot read exif data.")
            return {}
    else:
        print("Not a valid image file - cannot read exif data.")
        return {}

    ## populate dictionary
    exif_data_all = {}
    exif_data = {}
    try:
        for k, v in image._getexif().items():
            if k in ExifTags.TAGS:
                exif_data_all[ExifTags.TAGS[k]] = v
        exif_data_all = dict(sorted(exif_data_all.items()))
    except Exception:
        print("no meta-data found")
        return None

    if flag_show:
        print("--------------------------------------------")
        print("Available exif-tags:\n")
        pretty.pprint(exif_data_all)
        print("\n")
        print("Default exif-tags (append list using \"fields\" argument):\n")
        print(default_meta_data_fields)
        print("--------------------------------------------")

    ## subset exif_data
    for field in exif_fields:
        if field in exif_data_all:
            exif_data[field] = str(exif_data_all[field])
    exif_data = dict(sorted(exif_data.items()))

    ## close image and return meta data
    image.close()
    return exif_data



def show_image(image, max_dim=1200, position_reset=True, position_offset=25, 
               window_aspect="free"):
    """
    Show one or multiple images by providing path string or array or list of 
    either.
    
    Parameters
    ----------
    image: str, array, or list
        the image or list of images to be displayed. can be path to image 
        (string), array-type, or list of strings or arrays
    max_dim: int, optional
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
    """

    ## select window type
    if window_aspect == "free":
        window_aspect = cv2.WINDOW_NORMAL
    elif window_aspect == "fixed":
        window_aspect = cv2.WINDOW_AUTOSIZE

    
    ## open images list or single images
    while True:
        if isinstance(image, list):
            if len(image)>10:
                warning_banner = np.zeros((30,900,3))
                warning_string = "WARNING: trying to open " + str(len(image)) \
                    + " images - proceed (Enter) or stop (Esc)?"
                warning_image = cv2.putText(warning_banner,  
                                            warning_string ,(11,22), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.75, 
                                            colours["green"],
                                            1,
                                            cv2.LINE_AA)
                cv2.namedWindow('phenopype' ,cv2.WINDOW_AUTOSIZE )
                cv2.imshow('phenopype', warning_image)
                k = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if k == 27:
                    print("Stop - terminating.")
                    break
                elif k == 13:
                    print("Proceed - Opening images ...")
            idx=0
            for i in image:
                idx+=1
                _image_viewer(i, 
                              mode = "", 
                              window_aspect = window_aspect, 
                              window_name='phenopype' + " - " + str(idx), 
                              window_control="external",
                              max_dim=max_dim)
                if position_reset == True:
                    cv2.moveWindow('phenopype' + " - " + str(idx),
                                   idx + idx*position_offset,
                                   idx + idx*position_offset)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        else:
            _image_viewer(image, 
                  mode = "", 
                  window_aspect = window_aspect, 
                  window_name='phenopype', 
                  window_control="internal",
                  max_dim=max_dim)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break



def save_image(image, name, save_dir=os.getcwd(), resize=1, append="", 
               extension="jpg", overwrite=False, **kwargs):
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
    if "." in name:
        warnings.warn("need name and extension specified separately")
        return
    if append == "":
        append = ""
    else:
        append = "_" + append
    if "." not in extension:
        extension = "." + extension 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## resize
    if resize < 1:
        image = cv2.resize(image, 
                           (0,0), 
                           fx=1*resize, 
                           fy=1*resize, 
                           interpolation=cv2.INTER_AREA)

    ## construct save path
    new_name = name + append + extension
    path = os.path.join(save_dir, new_name)

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