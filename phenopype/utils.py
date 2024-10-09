#%% modules

import cv2
import io
import os
import sys
import webbrowser

from dataclasses import make_dataclass
from contextlib import redirect_stdout
from pkg_resources import resource_filename
import ruamel.yaml

from phenopype import _vars
from phenopype import config
from phenopype import decorators
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
                    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if image.shape[2] == 4:
                        b, g, r, a = cv2.split(image)
                        image = cv2.merge((r, g, b, a))
                    else:
                        b, g, r = cv2.split(image)
                        image = cv2.merge((r, g, b))
            else:
                ul._print(
                    'Invalid file extension "{}" - could not load image:\n'.format(ext)
                )
                return
        elif os.path.isdir(path):
            image = ul._load_project_image_directory(
                path, as_container=False
            )
        else:
            ul._print("Invalid image path - could not load image.")
            return
    else:
        ul._print("Invalid input format - could not load image.")
        return

    return image


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
        interpolation="nearest_exact",
        ):
    """
    Resize image by resize factor 

    Parameters
    ----------
    image: array 
        Image to be resized
    factor: float, optional
        Resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of original size). At 1, no resizing will be performed
    factor_ret: bool, optional
        If True, returns the image and the resize factor
    width: int, optional
        Width to resize the image to. If specified, height must also be specified
    height: int, optional
        Height to resize the image to. If specified, width must also be specified
    max_dim: int, optional
        Maximum size of any dimension that the image will be resized to. Maintains aspect ratio
    interpolation: {'nearest', 'linear', 'cubic', 'area', 'lanczos4', 'linear_exact', 
                    'nearest_exact', 'max', 'warp_fill_outliers', 'warp_inverse_map'} str, optional
        interpolation algorithm to use - refer to https://docs.opencv.org/4.9.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

    Returns
    -------
    image : ndarray
        resized image

    """
    
    image_height, image_width = image.shape[:2]

    if width is not None and height is not None:
        image = cv2.resize(
            image,
            (width, height),
            interpolation=_vars.opencv_interpolation_flags[interpolation]
        )
    elif max_dim is not None:
        if image_height > max_dim or image_width > max_dim:
            if image_width >= image_height:
                factor = max_dim / image_width
            else:
                factor = max_dim / image_height
            new_image_width = int(image_width * factor)
            new_image_height = int(image_height * factor)
            image = cv2.resize(
                image,
                (new_image_width, new_image_height),
                interpolation=_vars.opencv_interpolation_flags[interpolation]
            )
    elif factor != 1:
        image = cv2.resize(
            image,
            (0, 0),
            fx=factor,
            fy=factor,
            interpolation=_vars.opencv_interpolation_flags[interpolation]
        )

    if factor_ret:
        return image, factor
    return image

@decorators.legacy_args
def save_image(image, file_path, suffix=None, overwrite=False, **kwargs):
    """Save an image (array) to a specified format.

    Parameters
    ----------
    image : array
        Image to save.
    file_path : str
        Full path (including directory, base name, and extension) to save the image.
    suffix : str, optional
        Suffix to append to the image name to prevent overwriting.
    overwrite : bool, optional
        If True, overwrite existing files with the same name.

    Returns
    -------
    bool
        True if the image was saved successfully, False otherwise.
    """
    # Normalize file path and extract base name and extension
    dir_path, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    
    # Handle suffix
    suffix = f"_{suffix}" if suffix else ""
    
    # Construct new file path
    file_name_new = f"{base_name}{suffix}{ext}"
    new_file_path = os.path.join(dir_path, file_name_new)

    # Check if file exists and handle overwrite logic
    if ul._overwrite_check(new_file_path, overwrite):
        success = cv2.imwrite(new_file_path, image)
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
        ul._print("wrong input format.")
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
                    ul._print("Proceed - Opening images ...")
                    pass
                else:
                    ul._print("Aborting")
                    break
            idx = 0
            for i in image:
                idx += 1
                if i.__class__.__name__ == "ndarray":
                    ul._print("phenopype" + " - " + str(idx))
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
                    ul._print("skipped showing list item of type " + i.__class__.__name__)
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
