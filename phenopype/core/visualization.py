#%% modules
import ast, cv2, copy, os
import numpy as np
import pandas as pd
import math
import string

from phenopype.settings import AttrDict, colours, _annotation_function_dicts
from phenopype.utils_lowlevel import (
    _auto_line_width,
    _auto_point_size,
    _auto_text_width,
    _auto_text_size,
)

#%% settings

inf = math.inf

#%% functions


def draw_contours(
    image,
    annotation,
    offset_coords=None,
    line_colour="green",
    line_width="auto",
    fill=0.3,
    label=False,
    label_colour="black",
    label_font_size="auto",
    label_font_width="auto",
    bounding_box=False,
    bounding_box_ext=20,
    bounding_box_colour="black",
    bounding_box_line_width="auto",
    **kwargs,
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
    level = kwargs.get("level",3)
    fill_colour = kwargs.get("fill_colour", line_colour)

    flags = AttrDict({
        "bounding_box":bounding_box, 
        "label":label
        })
    
    if fill > 0:
        flags.fill = True
        fill_colour = colours[fill_colour]
    else:
        flags.fill = False

    line_colour = colours[line_colour]
    label_colour = colours[label_colour]
    bounding_box_colour_sel = colours[bounding_box_colour]
    
    ## more kwargs
    if line_width == "auto":
        line_width = _auto_line_width(image)
    if bounding_box_line_width == "auto":
        bounding_box_line_width = _auto_line_width(image)
    if label_font_size == "auto":
        label_size = _auto_text_size(image)
    if label_font_width == "auto":
        label_width = _auto_text_width(image)
                
    ## method
    canvas = copy.deepcopy(image)
       
    ## check annotation dict input and convert to type/id/ann structure
    if list(annotation.keys())[0] == "info":
        if annotation["info"]["annotation_type"] == "contour":
            contours = annotation["data"]["coord_list"]
            contours_support = annotation["data"]["support"]
            print("THIS")


    else:
        if not kwargs.get("contour_id"):
            print("- contour_id missing - please provide contour ID [a-z]")
            return
        else:
           contour_id = kwargs.get("contour_id")
        if list(annotation.keys())[0] in _annotation_function_dicts.keys():
            contours = annotation["contour"][contour_id]["data"]["coord_list"]
            contours_support = annotation["contour"][contour_id]["data"]["support"]
        elif list(annotation.keys())[0] in string.ascii_lowercase:
            contours = annotation[contour_id]["data"]["coord_list"]
            contours_support = annotation[contour_id]["data"]["support"]
       
    
    ## 1) fill contours
    if flags.fill:
        colour_mask = copy.deepcopy(canvas)
        for contour in contours:
            cv2.drawContours(
                image=canvas,
                contours=[contour],
                contourIdx=0,
                thickness=-1,
                color=fill_colour,
                maxLevel=level,
                offset=offset_coords,
                )
        canvas = cv2.addWeighted(colour_mask, 1 - fill, canvas, fill, 0)

    ## 2) contour lines
    for contour in contours:
        cv2.drawContours(
            image=canvas,
            contours=[contour],
            contourIdx=0,
            thickness=line_width,
            color=line_colour,
            maxLevel=level,
            offset=offset_coords,
        )

    ## 3) bounding boxes
    if flags.bounding_box:
        q = bounding_box_ext
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            cv2.rectangle(
                canvas,
                (rx - q, ry - q),
                (rx + rw + q, ry + rh + q),
                bounding_box_colour_sel,
                bounding_box_line_width,
            )

    ## 4) contour label 
    if flags.label:
        for idx, support in enumerate(contours_support):
            cv2.putText(
            canvas,
            str(idx),
            tuple(support["center"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_size,
            label_colour,
            label_width,
            cv2.LINE_AA,
        )

    ## return
    return canvas

    # ## load previous contours
    # if obj_input.__class__.__name__ == "container" and not compare.__class__.__name__ == "NoneType":
    #     while True:
    #         if compare.__class__.__name__ == "str":
    #             compare = [compare]
    #         elif compare.__class__.__name__ == "list" and len(compare) > 3:
    #             print("compare supports a maximum of three contour files")
    #             break
    #         col_idx = 0
    #         cols = ["green","blue","red","black","white"]
    #         for comp in compare:
    #             if line_colour == cols[col_idx]:
    #                 col_idx +=1
    #             comp_line_colour = colours[cols[col_idx]]
    #             comp_path = os.path.join(obj_input.dirpath, "contours_" + comp + ".csv")
    #             col_idx +=1
    #             if os.path.isfile(comp_path):
    #                 compare_df = pd.read_csv(comp_path, converters={"center": ast.literal_eval})
    #                 if "x" in compare_df:
    #                     compare_df["coords"] = list(zip(compare_df.x, compare_df.y))
    #                     coords = compare_df.groupby("contour")["coords"].apply(list)
    #                     coords_arr = _contours_tup_array(coords)
    #                     compare_df.drop(columns=["coords", "x", "y"], inplace=True)
    #                     compare_df = compare_df.drop_duplicates().reset_index()
    #                     compare_df["coords"] = pd.Series(coords_arr, index=compare_df.index)
    #                 else:
    #                     print("no coords found, cannot draw contours for comparison")
    #                     continue
    #                 print("- " + comp + " contours loaded")           
    #                 for key, value in contours_compare.items():
    #                     cv2.drawContours(
    #                         image=image,
    #                         contours=[value["coords"]],
    #                         contourIdx=0,
    #                         thickness=compare_line_width,
    #                         color=comp_line_colour,
    #                         maxLevel=level,
    #                         offset=None,
    #                     )
    #             else:
    #                 print("wrong compare suffix")
    #         break

    # df_contours= df_contours.drop("skeleton_coords", axis=1)

    ## return
    return image


def draw_landmarks(
    obj_input,
    df_landmarks=None,
    label=True,
    label_colour="black",
    label_size="auto",
    label_width="auto",
    offset=0,
    point_colour="green",
    point_size="auto",
):
    """
    Draw landmarks into an image.

    Parameters
    ----------
    obj_input : array or container
        input object
    df_landmarks: DataFrame, optional
        should contain landmark coordinates as an array in a df cell
    label : bool, optional
        draw landmark label
    label_colour : {"black", "white", "green", "red", "blue"} str, optional
        landmark label colour.
    label_size: int, optional
        landmark label font size (scaled to image)
    label_width: int, optional
        landmark label font width  (scaled to image)
    offset: int, optional
        add offset (in pixels) to text location (to bottom-left corner of the text string)
    point_colour: {"green", "red", "blue", "black", "white"} str, optional
        landmark point colour
    point_size: int, optional
        landmark point size in pixels

    Returns
    -------
    image: array or container
        image with landmarks

    """

    ## kwargs
    point_colour = colours[point_colour]
    label_col = colours[label_colour]
    flag_label = label

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
        if flag_label:
            x,y = x+offset, y+offset
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
    masks,
    line_colour="blue",
    line_width="auto",
    label=False,
    label_size="auto",
    label_colour="black",
    label_width="auto",
):
    """
    Draw masks into an image. This function is also used to draw the perimeter 
    of a created or detected reference card.
    
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
    line_colour = colours[line_colour]
    label_colour = colours[label_colour]

    coords = masks["data"]["coords"]

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
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
    for coord in coords:
        if coord[0].__class__.__name__ == "list":
            coord = coord[0]
        cv2.polylines(image, np.array([coord]), False, line_colour, line_width)
        if flag_label:
            cv2.putText(
                image,
                label,
                coords[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                label_colour,
                label_width,
                cv2.LINE_AA,
            )


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



def select_canvas(container, canvas="raw", multi_channel=True):
    """
    Isolate a colour channel from an image or select canvas for the pype method.

    Parameters
    ----------

    canvas : {"mod", "bin", "gray", "raw", "red", "green", "blue"} str, optional
        the type of canvas to be used for visual feedback. some types require a
        function to be run first, e.g. "bin" needs a segmentation algorithm to be
        run first. black/white images don't have colour channels. coerced to 3D
        array by default
    multi: bool, optional
        coerce returned array to multichannel (3-channel)

    Returns
    -------
    obj_input : container
        canvas can be called with "obj_input.canvas".

    """

    ## method
    if canvas == "mod":
        container.canvas = copy.deepcopy(container.image)
        print("- modifed image")
    elif canvas == "raw":
        container.canvas = copy.deepcopy(container.image_copy)
        print("- raw image")
    elif canvas == "bin":
        container.canvas = copy.deepcopy(container.image_bin)
        print("- binary image")
    elif canvas == "gray":
        container.canvas = cv2.cvtColor(container.image, cv2.COLOR_BGR2GRAY)
        print("- grayscale image")
    elif canvas == "green":
        container.canvas = container.image[:, :, 0]
        print("- green channel")
    elif canvas == "red":
        container.canvas = container.image[:, :, 1]
        print("- red channel")
    elif canvas == "blue":
        container.canvas = container.image[:, :, 2]
        print("- blue channel")
    else:
        print("- invalid selection - defaulting to raw image")
        container.canvas = copy.deepcopy(container.image_copy)

    ## check if colour
    if multi_channel:
        if len(container.canvas.shape) < 3:
            container.canvas = cv2.cvtColor(container.canvas, cv2.COLOR_GRAY2BGR)
            
            
        