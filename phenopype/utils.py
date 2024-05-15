#%% modules

import cv2
import io
import os
import sys
import webbrowser

from datetime import datetime
from dataclasses import make_dataclass
from contextlib import redirect_stdout
from pkg_resources import resource_filename
import ruamel.yaml

from phenopype import _vars
from phenopype import config
from phenopype import utils_lowlevel as ul


#%% functions


def load_image(path, mode="unchanged", **kwargs):
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
            if ext.replace(".", "") in _vars.default_filetypes:
                if flags.mode == "unchanged":
                    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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
            image = ul._load_project_image_directory(
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
    if not config.template_path_current == template_path:

        if template_path.__class__.__name__ == "str":
            if os.path.isfile(template_path):
                template_loaded = ul._load_yaml(template_path)
                config.template_path_current = template_path
                config.template_loaded_current = ul._load_yaml(template_path)

            else:
                print("Could not find template_path")
                return
        else:
            print("Wrong input format for template_path")
            return
    else:
        template_loaded =  config.template_loaded_current

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
            "date_created": datetime.today().strftime(_vars.strftime_format),
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
        ul._yaml_recursive_delete_comments(template_loaded)

    if ul._save_prompt("template", config_path, flags.overwrite):
        with open(config_path, "wb") as yaml_file:
            yaml.dump(template_loaded, yaml_file)

    if ret_path:
        return config_path



def print_colours():

    colours_path = os.path.join(resource_filename("phenopype", "assets"), "wc3_colours.html")
    webbrowser.open_new_tab(colours_path)


def resize_image(
        image, 
        factor=1, 
        factor_ret=False,
        width=None,
        height=None,
        max_dim=None, 
        interpolation="cubic",
        ):
    """
    Resize image by resize factor 

    Parameters
    ----------
    image: array 
        image to be resized
    max_dim: int, optional
        maximum size of any dimension that the image will be resized to. if 
        image is smaller, no resizing will be performed. maintains aspect ratio
    factor: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of 
        original size). at 1, no resizing will be performed
    interpolation: {'nearest', 'linear', 'cubic', 'area', 'lanczos', 'lin_exact', 'inter', 'warp_fill', 'warp_inverse'} str, optional
        interpolation algorithm to use - refer to https://docs.opencv.org/3.4.9/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

    Returns
    -------
    image : ndarray
        resized image

    """
    image_height, image_width = image.shape[0:2]

    ## method
    if not all([
            width.__class__.__name__ == "NoneType",
            height.__class__.__name__ == "NoneType",
            ]):
        image = cv2.resize(
            image,
            (width, height),
            interpolation=_vars.opencv_interpolation_flags[interpolation],
        )
    
    elif not max_dim.__class__.__name__ == "NoneType":
        if image_height > max_dim or image_width > max_dim:
            if image_width >= image_height:
                factor = max_dim / image_width
                new_image_width, new_image_height = (
                    max_dim,
                    int(factor * image_height),
                )
            elif image_height > image_width:
                factor = max_dim / image_height
                new_image_width, new_image_height = (
                    int(factor * image_width),
                    max_dim,
                )
            image = cv2.resize(
                image,
                (new_image_width, new_image_height),
                interpolation=_vars.opencv_interpolation_flags[interpolation],
            )
            
    else:        
        if factor == 1:
            pass
        else:
            image = cv2.resize(
                image,
                (0, 0),
                fx=1 * factor,
                fy=1 * factor,
                interpolation=_vars.opencv_interpolation_flags[interpolation],
            )

    ## return results
    if factor_ret:
        return image, factor    
    else:
        return image


def save_image(image, file_name, dir_path, suffix=None, ext="jpg", overwrite=False):
    """Save an image (array) to a specified format.

    Parameters
    ----------
    image : array
        Image to save.
    file_name : str
        Base name for the saved image.
    dir_path : str
        Directory to save the image.
    suffix : str, optional
        Suffix to append to the image name to prevent overwriting.
    ext : str, optional
        File extension to save image as.
    overwrite : bool, optional
        If True, overwrite existing files with the same name.

    Returns
    -------
    bool
        True if the image was saved successfully, False otherwise.
    """
    # Normalize file extension
    if not ext.startswith("."):
        ext = "." + ext

    # Handle suffix
    suffix = f"_{suffix}" if suffix else ""

    # Construct full file path
    base_name, original_ext = os.path.splitext(file_name)
    if original_ext:
        file_name = base_name  # Remove original extension if it exists
    file_name_new = f"{file_name}{suffix}{ext}"
    file_path = os.path.join(dir_path, file_name_new)

    # Check if file exists and handle overwrite logic
    if os.path.isfile(file_path) and not overwrite:
        print(f"Image not saved - file already exists (overwrite=False): {file_path}")
        return False
    else:
        if overwrite and os.path.isfile(file_path):
            print(f"Image saved and overwritten at: {file_path}")
        else:
            print(f"Image saved at: {file_path}")

        # Save the image
        success = cv2.imwrite(file_path, image)
        return success



def show_image(
    image,
    return_input=False,
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
                    ul._GUI(
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
                            int(idx + idx * position_offset),
                            int(idx + idx * position_offset),
                        )
                else:
                    print("skipped showing list item of type " + i.__class__.__name__)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        else:
            out = ul._GUI(
                image=image,
                mode="",
                return_input=return_input,
                window_aspect=window_aspect,
                window_name="phenopype",
                window_control="internal",
                **kwargs,
            )
            if return_input:
                return out.keypress_trans
            break
