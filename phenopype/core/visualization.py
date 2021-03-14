#%% modules
import ast, cv2, copy, os
import numpy as np
import pandas as pd
import math

from phenopype.settings import colours
from phenopype.utils_lowlevel import (
    _auto_line_width,
    _auto_point_size,
    _auto_text_width,
    _auto_text_size,
    _contours_tup_array,
)

#%% settings

inf = math.inf

#%% functions


def select_canvas(obj_input, canvas="image_mod", multi=True):
    """
    Isolate a colour channel from an image or select canvas for the pype method.

    Parameters
    ----------
    obj_input : array or container
        input object
    canvas : {"mod", "bin", "gray", "raw", "red", "green", "blue"} str, optional
        the type of canvas to be used for visual feedback. some types require a
        function to be run first, e.g. "bin" needs a segmentation algorithm to be
        run first. black/white images don't have colour channels. coerced to 3D
        array by default
    multi: bool, optional
        coerce returned array to multilevel (3-level)

    Returns
    -------
    obj_input : container
        canvas can be called with "obj_input.canvas".

    """
    ## kwargs
    flag_multi = multi

    ## load image
    flag_container = False
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        flag_container = True
        image = copy.deepcopy(obj_input.image_copy)
    else:
        print("wrong input format.")
        return

    ## method
    ## for container
    if flag_container and canvas in ["image_bin", "bin", "binary"]:
        canvas = copy.deepcopy(obj_input.image_bin)
        print("- binary image")
    elif flag_container and canvas in ["image_mod", "mod", "modified"]:
        canvas = copy.deepcopy(obj_input.image)
        print("- modifed image")
    elif flag_container and canvas in ["image_raw", "raw"]:
        canvas = copy.deepcopy(obj_input.image_copy)
        print("- raw image")
    ## for array and container
    elif canvas in ["gray", "grey", "image_gray"]:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("- grayscale image")
    elif canvas in ["g", "green"]:
        canvas = image[:, :, 0]
        print("- green channel")
    elif canvas in ["r", "red"]:
        canvas = image[:, :, 1]
        print("- red channel")
    elif canvas in ["b", "blue"]:
        canvas = image[:, :, 2]
        print("- blue channel")
    else:
        print("- invalid selection - defaulting to raw image")
        canvas = copy.deepcopy(obj_input.image_copy)

    ## check if 3D
    if flag_multi:
        if len(canvas.shape) < 3:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return canvas
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = canvas


def draw_contours(
    obj_input,
    df_contours=None,
    offset_coords=None,
    compare=None,
    compare_line_width=1,
    label=True,
    fill=0.3,
    fill_colour=None,
    mark_holes=True,
    level=3,
    line_colour="green",
    label_size="auto",
    label_colour="black",
    line_width="auto",
    label_width="auto",
    skeleton=True,
    watershed=False,
    bounding_box=False,
    bounding_box_ext=20,
    bounding_box_colour="black",
    bounding_box_line_width="auto"
):
    """
    Draw contours and their labels onto a canvas. Can be filled or empty, offset
    coordinates can be supplied. This will also draw the skeleton, if the argument
    "skeleton=True" and the supplied "df_contour" contains a "skeleton_coords"
    column.

    Parameters
    ----------
    obj_input : array or container
        input object
    df_contours : DataFrame, optional
        contains the contours
    offset_coords : tuple, optional
        offset coordinates, will be added to all contours
    compare: str or list, optional
        draw previously detected contours as well (e.g. from other pype-run). 
        string or list of strings with file-suffixes, has to be in the same 
        directory.
    label : bool, optional
        draw contour label
    fill : float, optional
        background transparency for contour fill (0=no fill).
    fill_colour : {"green", "red", "blue", "black", "white"} str, optional
        contour fill colour - if not specified, defaults to line colour
    mark_holes : bool, optional
        contours located inside other contours (i.e. their holes) will be 
        highlighted in red
    level : int, optional
        the default is 3.
    line_colour: {"green", "red", "blue", "black", "white"} str, optional
        contour line colour
    line_width: int, optional
        contour line width
    label_colour : {"black", "white", "green", "red", "blue"} str, optional
        contour label colour.
    label_size: int, optional
        contour label font size (scaled to image)
    label_width: int, optional
        contour label font thickness 
    watershed: bool, optional
        indicates if a watershed-procedure has been performed. formats the
        coordinate colours accordingly (excludes "mark_holes option")
    bounding_box: bool, optional
        draw bounding box around the contour
    bounding_box_ext: in, optional
        value in pixels by which the bounding box should be extended
    bounding_box_colour: {"green", "red", "blue", "black", "white"} str, optional
        bounding box line colour
    bounding_box_line_width: int, optional
        bounding box line width
        
    Returns
    -------
    image: array or container
        image with contours

    """
    ## kwargs
    flag_bounding_box = bounding_box
    if flag_bounding_box:
        q = bounding_box_ext
    flag_label = label
    flag_fill = fill
    flag_mark_holes = mark_holes
    flag_skeleton = skeleton
    flag_watershed = watershed
    if flag_watershed:
        flag_mark_holes = True
    line_colour_sel = colours[line_colour]
    label_colour = colours[label_colour]
    bounding_box_colour_sel = colours[bounding_box_colour]
    
    if fill_colour.__class__.__name__ == "NoneType":
        fill_colour = line_colour_sel
    else:
        fill_colour = colours[fill_colour]
        
    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
        if df_contours.__class__.__name__ == "NoneType":
            print("No df provided - cannot draw contours.")
            return
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_contours = obj_input.df_contours
    else:
        print("wrong input format.")
        return
    
    ## more kwargs
    if line_width == "auto":
        line_width = _auto_line_width(image)
    if bounding_box_line_width == "auto":
        bounding_box_line_width = _auto_line_width(image)
    if label_size == "auto":
        label_size = _auto_text_size(image)
    if label_width == "auto":
        label_width = _auto_text_width(image)
        
    ## method
    idx = 0
    colour_mask = copy.deepcopy(image)
    for index, row in df_contours.iterrows():
        if flag_mark_holes:
            if row["order"] == "child":
                if flag_watershed:
                    line_colour = line_colour_sel
                else:
                    line_colour = colours["red"]
            elif row["order"] == "parent":
                line_colour = line_colour_sel
        else:
            line_colour = line_colour_sel
        if flag_fill > 0:
            cv2.drawContours(
                image=colour_mask,
                contours=[row["coords"]],
                contourIdx=idx,
                thickness=-1,
                color=fill_colour,
                maxLevel=level,
                offset=offset_coords,
            )
        if line_width > 0:
            cv2.drawContours(
                image=image,
                contours=[row["coords"]],
                contourIdx=idx,
                thickness=line_width,
                color=line_colour,
                maxLevel=level,
                offset=offset_coords,
            )
        if flag_skeleton and "skeleton_coords" in df_contours:
            cv2.drawContours(
                image=image,
                contours=[row["skeleton_coords"]],
                contourIdx=idx,
                thickness=line_width,
                color=colours["red"],
                maxLevel=level,
                offset=offset_coords,
            )
        if flag_label:
            cv2.putText(
                image,
                str(row["contour"]),
                (row["center"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                label_colour,
                label_width,
                cv2.LINE_AA,
            )
            cv2.putText(
                colour_mask,
                str(row["contour"]),
                (row["center"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                label_colour,
                label_width,
                cv2.LINE_AA,
            )

    image = cv2.addWeighted(image, 1 - flag_fill, colour_mask, flag_fill, 0)
    
    for index, row in df_contours.iterrows():
        if flag_bounding_box:
            rx, ry, rw, rh = cv2.boundingRect(row["coords"])
            cv2.rectangle(
                image,
                (rx - q, ry - q),
                (rx + rw + q, ry + rh + q),
                bounding_box_colour_sel,
                bounding_box_line_width,
            )

    ## load previous contours
    if obj_input.__class__.__name__ == "container" and not compare.__class__.__name__ == "NoneType":
        while True:
            if compare.__class__.__name__ == "str":
                compare = [compare]
            elif compare.__class__.__name__ == "list" and len(compare) > 3:
                print("compare supports a maximum of three contour files")
                break
            col_idx = 0
            cols = ["green","blue","red","black","white"]
            for comp in compare:
                if line_colour == cols[col_idx]:
                    col_idx +=1
                comp_line_colour = colours[cols[col_idx]]
                comp_path = os.path.join(obj_input.dirpath, "contours_" + comp + ".csv")
                col_idx +=1
                if os.path.isfile(comp_path):
                    compare_df = pd.read_csv(comp_path, converters={"center": ast.literal_eval})
                    if "x" in compare_df:
                        compare_df["coords"] = list(zip(compare_df.x, compare_df.y))
                        coords = compare_df.groupby("contour")["coords"].apply(list)
                        coords_arr = _contours_tup_array(coords)
                        compare_df.drop(columns=["coords", "x", "y"], inplace=True)
                        compare_df = compare_df.drop_duplicates().reset_index()
                        compare_df["coords"] = pd.Series(coords_arr, index=compare_df.index)
                    else:
                        print("no coords found, cannot draw contours for comparison")
                        continue
                    print("- " + comp + " contours loaded")           
                    for index, row in compare_df.iterrows():
                        cv2.drawContours(
                            image=image,
                            contours=[row["coords"]],
                            contourIdx=0,
                            thickness=compare_line_width,
                            color=comp_line_colour,
                            maxLevel=level,
                            offset=None,
                        )
                else:
                    print("wrong compare suffix")
            break  

    # df_contours= df_contours.drop("skeleton_coords", axis=1)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = image


def draw_landmarks(
    obj_input,
    df_landmarks=None,
    point_colour="green",
    point_size="auto",
    label_colour="black",
    label_size="auto",
    label_width="auto",
):
    """
    Draw landmarks into an image.

    Parameters
    ----------
    obj_input : array or container
        input object
    df_landmarks: DataFrame, optional
        should contain landmark coordinates as an array in a df cell
    point_colour: {"green", "red", "blue", "black", "white"} str, optional
        landmark point colour
    point_size: int, optional
        landmark point size in pixels
    label_colour : {"black", "white", "green", "red", "blue"} str, optional
        landmark label colour.
    label_size: int, optional
        landmark label font size (scaled to image)
    label_width: int, optional
        landmark label font width  (scaled to image)

    Returns
    -------
    image: array or container
        image with landmarks

    """

    ## kwargs
    point_colour = colours[point_colour]
    label_col = colours[label_colour]

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_landmarks.__class__.__name__ == "NoneType":
            print("No df provided - cannot draw landmarks.")
            return
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_landmarks = obj_input.df_landmarks
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if point_size == "auto":
        point_size = _auto_point_size(image)
    if label_size == "auto":
        label_size = _auto_text_size(image)
    if label_width == "auto":
        label_width = _auto_text_width(image)

    ## visualize
    for label, x, y in zip(df_landmarks.landmark, df_landmarks.x, df_landmarks.y):
        cv2.circle(image, (x, y), point_size, point_colour, -1)
        cv2.putText(
            image,
            str(label),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_size,
            label_col,
            label_width,
            cv2.LINE_AA,
        )

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = image


def draw_masks(
    obj_input,
    select=None,
    df_masks=None,
    line_colour="blue",
    line_width="auto",
    label=False,
    label_size="auto",
    label_colour="black",
    label_width="auto",
):
    """
    Draw masks into an image. This function is also used to draw the perimeter 
    of a created or detected reference scale card.
    
    Parameters
    ----------        
    obj_input : array or container
        input object
    select: str or list
        select a subset of masks to display
    df_masks: DataFrame, optional
        contains mask coordinates and label
    line_colour: {"blue", "red", "green", "black", "white"} str, optional
        mask line colour
    line_width: int, optional
        mask line width
    label_colour : {"black", "white", "green", "red", "blue"} str, optional
        mask label colour.
    label_size: int, optional
        mask label font size (scaled to image)
    label_width: int, optional
        mask label font width  (scaled to image)

    Returns
    -------
    image: array or container
        image with masks
    """
    ## kwargs
    flag_label = label
    line_colour_sel = colours[line_colour]
    label_colour = colours[label_colour]
    if not select.__class__.__name__ == "NoneType":
        if not select.__class__.__name__ == "list":
            select = [select]

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
        if df_masks.__class__.__name__ == "NoneType":
            print("No df provided - cannot draw masks.")
            return
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_masks = obj_input.df_masks
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if line_width == "auto":
        line_width = _auto_line_width(image)
    if label_size == "auto":
        label_size = _auto_text_size(image)
    if label_width == "auto":
        label_width = _auto_text_width(image)

    ## draw masks from mask obect
    for index, row in df_masks.iterrows():
        if not select.__class__.__name__ == "NoneType":
            if row["mask"] in select:
                pass
            else:
                continue
        else:
            pass
        coords = eval(row["coords"])
        if coords[0].__class__.__name__ == "list":
            coords = coords[0]
        if row["mask"] == "scale":
            line_colour = colours["red"]
        else:
            line_colour = line_colour_sel
        cv2.polylines(image, np.array([coords]), False, line_colour, line_width)
        if flag_label:
            cv2.putText(
                image,
                row["mask"],
                coords[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                label_colour,
                label_width,
                cv2.LINE_AA,
            )

    for mask in df_masks["mask"].unique():
        print("drawing mask: " + str(mask)) 

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = image


def draw_polylines(
        obj_input, 
        df_polylines=None, 
        line_colour="blue", 
        line_width="auto",
        ):
    """
    Draw polylines onto an image. 
    
    Parameters
    ----------
    obj_input : array or container
        input object
    df_polylines: DataFrame, optional
        should contain polyline coordinates as an array in a df cell
    line_colour: {"blue", "red", "green", "black", "white"} str, optional
        polyline colour
    line_width: int, optional
        polyline width

    Returns
    -------
    image: array or container
        image with polylines
    """

    ## kwargs
    line_colour = colours[line_colour]

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_polylines.__class__.__name__ == "NoneType":
            print("No df provided - cannot draw landmarks.")
            return
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_polylines = obj_input.df_polylines
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if line_width == "auto":
        line_width = _auto_line_width(image)

    ## visualize
    for polyline in df_polylines["polyline"].unique():
        sub = df_polylines.groupby(["polyline"])
        sub = sub.get_group(polyline)
        coords = list(sub[["x", "y"]].itertuples(index=False, name=None))
        cv2.polylines(image, np.array([coords]), False, line_colour, line_width)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.canvas = image
