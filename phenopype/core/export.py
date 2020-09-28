#%% modules
import cv2, copy, os
import pandas as pd

from phenopype.utils_lowlevel import _save_yaml, _load_yaml, _contours_arr_tup

#%% functions


def save_canvas(
    obj_input, overwrite=True, dirpath=None, save_suffix=None, name="", resize=0.5
):
    """
    Save a canvas (processed image). 

    Parameters
    ----------
    obj_input : array or container
        input object
    overwrite : bool, optional
        overwrite flag in case file exists
    dirpath : str, optional
        folder to save file in 
    save_suffix : str, optional
        suffix to append to filename
    name: str, optional
        custom name for file
    resize: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of
        original size).

    """
    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.canvas)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No image supplied - cannot save canvas.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## resize
    if resize < 1:
        image = cv2.resize(image, (0, 0), fx=1 * resize, fy=1 * resize)

    ## save suffix
    if len(name) > 0:
        name = "_" + name
    if save_suffix:
        path = os.path.join(dirpath, "canvas" + name + "_" + save_suffix + ".jpg")
    else:
        path = os.path.join(dirpath, "canvas" + name + ".jpg")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- canvas not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- canvas saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- canvas saved under " + path + ".")
            pass
        cv2.imwrite(path, image)
        break


def save_colours(
    obj_input, overwrite=True, dirpath=None, save_suffix=None, round_digits=1
):
    """
    Save colour intensities to csv. 
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    round_digits: int, optional
        number of digits to round to
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_colours)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## fix na, round, and format to string
    df = df.fillna(-9999)
    df = df.round(round_digits)
    df = df.astype(str)

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "colours_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "colours.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- colours not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- colours saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- colours saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


def save_contours(
    obj_input,
    overwrite=True,
    dirpath=None,
    save_suffix=None,
    save_coords=False,
    convert_coords=True,
    subset=None,
    # round_digits=4
):
    """
    Save contour coordinates and features to csv. This also saves skeletonization
    ouput if the data is contained in the provided DataFrame.
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename
    save_coords: bool, optional
        save the contour coordinates
    convert_coords: bool, optional
        convert the coordinates from array to x and y column
    subset: {"parent", "child"} str, optional
        save only a subset of the parent-child order structure
    round_digits: int, optional
        number of digits to round to
    """

    ## kwargs
    flag_overwrite = overwrite
    flag_convert_coords = convert_coords
    flag_save_coords = save_coords
    flag_subset = subset

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_contours)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No contour df supplied - cannot export contours.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## drop skeleton coordinates
    if "skeleton_coords" in df:
        df.drop(columns="skeleton_coords", inplace=True)

    ## subset
    if flag_subset == "child":
        df = df[df["order"] == "child"]
    elif flag_subset == "parent":
        df = df[df["order"] == "parent"]

    ## convert contour coords to list of tuples
    if flag_convert_coords and flag_save_coords:
        for idx, row in df.iterrows():
            df.at[idx, "coords"] = _contours_arr_tup(row["coords"])
        df = df.explode("coords")
        df = pd.concat(
            [
                df.reset_index(drop=True),
                pd.DataFrame(df["coords"].tolist(), columns=["x", "y"]),
            ],
            axis=1,
        )
    else:
        df.drop(columns=["coords"], inplace=True)
        
    ## fix na, round, and format to string
    # df = df.fillna(-9999)
    # df = df.round(round_digits)
    # df = df.astype(str)

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "contours_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "contours.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- contours not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- contours saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- contours saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


# def save_drawing(obj_input, overwrite=True, dirpath=None):
#     """
#     Save drawing coordinates to attributes.yaml 
    
#     Parameters
#     ----------
#     obj_input : DataFrame or container
#         input object
#     overwrite: bool optional
#         overwrite if drawing already exists
#     dirpath: str, optional
#         location to save df

#     """
#     ## kwargs
#     flag_overwrite = overwrite

#     ## load df
#     if obj_input.__class__.__name__ == "DataFrame":
#         df = obj_input
#     elif obj_input.__class__.__name__ == "container":
#         df = obj_input.df_draw
#         if (
#             dirpath.__class__.__name__ == "NoneType"
#             and not obj_input.dirpath.__class__.__name__ == "NoneType"
#         ):
#             dirpath = obj_input.dirpath
#     else:
#         print("No df supplied - cannot export results.")
#         return

#     ## dirpath
#     if dirpath.__class__.__name__ == "NoneType":
#         print('No save directory ("dirpath") specified - cannot save result.')
#         return
#     else:
#         if not os.path.isdir(dirpath):
#             q = input("Save folder {} does not exist - create?.".format(dirpath))
#             if q in ["True", "true", "y", "yes"]:
#                 os.makedirs(dirpath)
#             else:
#                 print("Directory not created - aborting")
#                 return

#     ## load attributes file
#     attr_path = os.path.join(dirpath, "attributes.yaml")
#     if os.path.isfile(attr_path):
#         attr = _load_yaml(attr_path)
#     else:
#         attr = {}
#     if not "drawing" in attr:
#         attr["drawing"] = {}

#     ## check if file exists
#     while True:
#         if "drawing" in attr and flag_overwrite == False:
#             print("- drawing not saved (overwrite=False)")
#             break
#         elif "drawing" in attr and flag_overwrite == True:
#             print("- drawing saved (overwriting)")
#             pass
#         elif not "drawing" in attr:
#             attr["drawing"] = {}
#             print("- drawing saved")
#             pass
#         for idx, row in df.iterrows():
#             # if not row["coords"] == attr["drawing"]["coords"]:
#             attr["drawing"] = dict(row)
#         _save_yaml(attr, attr_path)
#         break
    
def save_drawings(obj_input, overwrite=True, dirpath=None, save_suffix=None):
    """
    Save drawing coordinates and information ("include"" and "label") to csv.

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename

    """
    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "ndarray":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_drawings
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No drawing df supplied - cannot save drawing.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "drawings_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "drawings.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- drawings not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- drawings saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- drawings saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break



def save_data_entry(obj_input, overwrite=True, dirpath=None):
    """
    Save data entry to attributes.yaml 

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite if entry already exists
    dirpath: str, optional
        location to save df
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_other_data
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
    else:
        print("No df supplied - cannot export results.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## load attributes file
    attr_path = os.path.join(dirpath, "attributes.yaml")
    if os.path.isfile(attr_path):
        attr = _load_yaml(attr_path)
    else:
        attr = {}
    if not "other" in attr:
        attr["other"] = {}

    ## check if entry exists
    while True:
        for col in list(df):
            if col in attr["other"] and flag_overwrite == False:
                print("- column " + col + " not saved (overwrite=False)")
                continue
            elif col in attr["other"] and flag_overwrite == True:
                print("- add column " + col + " (overwriting)")
                pass
            else:
                print("- add column " + col)
                pass
            attr["other"][col] = df[col][0]
        break
    _save_yaml(attr, attr_path)


def save_landmarks(obj_input, overwrite=True, dirpath=None, save_suffix=None):
    """
    Save landmark coordinates to csv.

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_landmarks
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if not save_suffix.__class__.__name__ == "NoneType":
        path = os.path.join(dirpath, "landmarks_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "landmarks.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- landmarks not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- landmarks saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- landmarks saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


def save_masks(obj_input, overwrite=True, dirpath=None, save_suffix=None):
    """
    Save mask coordinates and information ("include"" and "label") to csv.

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename

    """
    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "ndarray":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_masks
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No mask df supplied - cannot save mask.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "masks_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "masks.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- masks not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- masks saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- masks saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


def save_polylines(obj_input, overwrite=True, dirpath=None, save_suffix=None):
    """
    Save polylines to csv.

    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = obj_input
    elif obj_input.__class__.__name__ == "container":
        df = obj_input.df_polylines
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "polylines_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "polylines.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- polylines not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- polylines saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- polylines saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False)
        break


def save_scale(obj_input, overwrite=True, dirpath=None):
    """
    Save a created or detected scale to attributes.yaml
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ in ["int", "float"]:
        scale_current_px_mm_ratio = obj_input
    elif obj_input.__class__.__name__ == "container":
        if hasattr(obj_input, "scale_current_px_mm_ratio"):
            scale_current_px_mm_ratio = obj_input.scale_current_px_mm_ratio
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
    else:
        print("No scale supplied - cannot export results.")
        return

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## load attributes file
    attr_path = os.path.join(dirpath, "attributes.yaml")
    if os.path.isfile(attr_path):
        attr = _load_yaml(attr_path)
    else:
        attr = {}
    if not "scale" in attr:
        attr["scale"] = {}

    ## check if file exists
    while True:
        if "current_px_mm_ratio" in attr["scale"] and flag_overwrite == False:
            print("- scale not saved (overwrite=False)")
            break
        elif "current_px_mm_ratio" in attr["scale"] and flag_overwrite == True:
            print("- save scale to attributes (overwriting)")
            pass
        else:
            print("- save scale to attributes")
            pass
        attr["scale"]["current_px_mm_ratio"] = scale_current_px_mm_ratio
        break
    _save_yaml(attr, attr_path)


def save_textures(
    obj_input, overwrite=True, dirpath=None, save_suffix=None, round_digits=1,
    name=""
):
    """
    Save colour intensities to csv. 
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    round_digits: int, optional
        number of digits to round to
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_textures)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## fix na, round, and format to string
    # df = df.fillna(-9999)
    df = df.round(round_digits)
    df = df.astype(str)

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if len(name) > 0:
        name = "_" + name
    if save_suffix:
        path = os.path.join(dirpath, "textures" + name + "_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "textures" + name + ".csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- textures not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- textures saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- textures saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False, na_rep='NA')
        break

def save_shapes(
    obj_input, overwrite=True, dirpath=None, save_suffix=None, round_digits=1
):
    """
    Save colour intensities to csv. 
    
    Parameters
    ----------
    obj_input : DataFrame or container
        input object
    overwrite: bool optional
        overwrite csv if it already exists
    dirpath: str, optional
        location to save df
    round_digits: int, optional
        number of digits to round to
    save_suffix : str, optional
        suffix to append to filename
    """

    ## kwargs
    flag_overwrite = overwrite

    ## load df
    if obj_input.__class__.__name__ == "DataFrame":
        df = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        df = copy.deepcopy(obj_input.df_shapes)
        if (
            dirpath.__class__.__name__ == "NoneType"
            and not obj_input.dirpath.__class__.__name__ == "NoneType"
        ):
            dirpath = obj_input.dirpath
        if (
            save_suffix.__class__.__name__ == "NoneType"
            and not obj_input.save_suffix.__class__.__name__ == "NoneType"
        ):
            save_suffix = obj_input.save_suffix
    else:
        print("No df supplied - cannot export results.")
        return

    ## fix na, round, and format to string
    # df = df.fillna(-9999)
    df = df.round(round_digits)
    df = df.astype(str)

    ## dirpath
    if dirpath.__class__.__name__ == "NoneType":
        print('No save directory ("dirpath") specified - cannot save result.')
        return
    else:
        if not os.path.isdir(dirpath):
            q = input("Save folder {} does not exist - create?.".format(dirpath))
            if q in ["True", "true", "y", "yes"]:
                os.makedirs(dirpath)
            else:
                print("Directory not created - aborting")
                return

    ## save suffix
    if save_suffix:
        path = os.path.join(dirpath, "shapes_" + save_suffix + ".csv")
    else:
        path = os.path.join(dirpath, "shapes.csv")

    ## check if file exists
    while True:
        if os.path.isfile(path) and flag_overwrite == False:
            print("- shapes not saved - file already exists (overwrite=False).")
            break
        elif os.path.isfile(path) and flag_overwrite == True:
            print("- shapes saved under " + path + " (overwritten).")
            pass
        elif not os.path.isfile(path):
            print("- shapes saved under " + path + ".")
            pass
        df.to_csv(path_or_buf=path, sep=",", index=False, na_rep='NA')
        break