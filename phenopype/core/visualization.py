#%% modules
import ast, cv2, copy, os
import numpy as np
import pandas as pd
import math
import string
from dataclasses import make_dataclass

from phenopype.settings import AttrDict, colours, _annotation_types
from phenopype.utils_lowlevel import (
    _auto_line_width,
    _auto_point_size,
    _auto_text_width,
    _auto_text_size,
    _provide_annotation_data,
)

#%% settings

inf = math.inf

#%% functions


def draw_contour(
    image,
    annotation,
    offset_coords=None,
    line_colour="green",
    line_width="auto",
    fill=0.3,
    label=False,
    label_colour="red",
    label_font_size="auto",
    label_font_width="auto",
    bounding_box=False,
    bounding_box_ext=20,
    bounding_box_colour="red",
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
	# =============================================================================
	# setup 

    ## kwargs
    level = kwargs.get("level",3)
    fill_colour = kwargs.get("fill_colour", line_colour)
    
    ## flags
    flags = make_dataclass(cls_name="flags", 
                            fields=[("bounding_box", bool, bounding_box), 
                                    ("label", str, label),
                                    ])
    
    ## filling and line settings
    if fill > 0:
        flags.fill = True
        fill_colour = colours[fill_colour]
    else:
        flags.fill = False

    line_colour = colours[line_colour]
    label_colour = colours[label_colour]
    bounding_box_colour_sel = colours[bounding_box_colour]
    
    if line_width == "auto":
        line_width = _auto_line_width(image)
    if bounding_box_line_width == "auto":
        bounding_box_line_width = _auto_line_width(image)
    if label_font_size == "auto":
        label_size = _auto_text_size(image)
    if label_font_width == "auto":
        label_width = _auto_text_width(image)


    ## extract annotation data     
    contours = _provide_annotation_data(annotation, "contour", "coord_list", kwargs)
    contours_support = _provide_annotation_data(annotation, "contour", "support", kwargs)

    if not contours or (not contours_support and flags.label):
        return image
    else:
        canvas = copy.deepcopy(image)
       
	# =============================================================================
	# execute
    
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
            str(idx + 1),
            tuple(support["center"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_size,
            label_colour,
            label_width,
            cv2.LINE_AA,
        )

    # =============================================================================
    # return
    
    return canvas



def draw_landmark(
    image,
    annotation,
    label=True,
    label_colour="black",
    label_size="auto",
    label_width="auto",
    offset=0,
    point_colour="green",
    point_size="auto",
    **kwargs
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

	# =============================================================================
	# setup 
    
    ## flags
    flags = make_dataclass(cls_name="flags", 
                            fields=[("label", bool, label)])
    
    ## point and label settings
    point_colour = colours[point_colour]
    label_col = colours[label_colour]

    if point_size == "auto":
        point_size = _auto_point_size(image)
    if label_size == "auto":
        label_size = _auto_text_size(image)
    if label_width == "auto":
        label_width = _auto_text_width(image)

    ## extract annotation data     
    points = _provide_annotation_data(annotation, "landmark", "points", kwargs)

    if not points:
        return image
    else:
        canvas = copy.deepcopy(image)


	# =============================================================================
	# execute
    
    for idx, point in enumerate(points):
        x,y = point
        cv2.circle(canvas, (x,y), point_size, point_colour, -1)
        if flags.label:
            x,y = x+offset, y+offset
            cv2.putText(
                canvas,
                str(idx + 1),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                label_col,
                label_width,
                cv2.LINE_AA,
            )
            

    # =============================================================================
    # return
    
    return canvas

def draw_mask(
    image,
    annotation,
    line_colour="blue",
    line_width="auto",
    label=False,
    label_size="auto",
    label_colour="black",
    label_width="auto",
    **kwargs
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
        image with coord_list
    """
	# =============================================================================
	# setup 

    ## flags
    flags = make_dataclass(cls_name="flags", 
                            fields=[("label", bool, label)])

    ## filling and line settings
    line_colour = colours[line_colour]
    label_colour = colours[label_colour]
    
    if line_width == "auto":
        line_width = _auto_line_width(image)
    if label_size == "auto":
        label_size = _auto_text_size(image)
    if label_width == "auto":
        label_width = _auto_text_width(image)
               
    ## extract annotation data     
    coord_list = _provide_annotation_data(annotation, "mask", "coord_list", kwargs)
    if not coord_list:
        return image
    else:
        canvas = copy.deepcopy(image)

	# =============================================================================
	# execute
    
    ## draw masks
    for coords in coord_list:
        cv2.polylines(canvas, np.array([coords]), False, line_colour, line_width)
        if flags.label:
            cv2.putText(
                canvas,
                label,
                coords[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                label_colour,
                label_width,
                cv2.LINE_AA,
            )


	# =============================================================================
	# return

    return canvas



def draw_polyline(
    image,
    annotation,
    line_colour="blue",
    line_width="auto",
    **kwargs
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
        image with coord_list
    """
	# =============================================================================
	# setup 

    ## line settings
    line_colour = colours[line_colour]
    
    if line_width == "auto":
        line_width = _auto_line_width(image)

    ## extract annotation data     
    coord_list = _provide_annotation_data(annotation, "line", "coord_list", kwargs)
    if not coord_list:
        return image
    else:
        canvas = copy.deepcopy(image)

    
	# =============================================================================
	# execute
    
    ## draw lines
    for coords in coord_list:
        cv2.polylines(canvas, np.array([coords]), False, line_colour, line_width)


	# =============================================================================
	# return

    return canvas



def select_canvas(container, canvas="raw", multi_channel=True, **kwargs):
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
    # elif canvas == "bin":
    #     container.canvas = copy.deepcopy(container.image_bin)
        # print("- binary image")
    elif canvas == "gray":
        container.canvas = cv2.cvtColor(container.image_gray, cv2.COLOR_BGR2GRAY)
        print("- grayscale image")
    elif canvas == "green":
        container.canvas = container.image_copy[:, :, 0]
        print("- green channel")
    elif canvas == "red":
        container.canvas = container.image_copy[:, :, 1]
        print("- red channel")
    elif canvas == "blue":
        container.canvas = container.image_copy[:, :, 2]
        print("- blue channel")
    else:
        print("- invalid selection - defaulting to raw image")
        container.canvas = copy.deepcopy(container.image_copy)

    ## check if colour
    if multi_channel:
        if len(container.canvas.shape) < 3:
            container.canvas = cv2.cvtColor(container.canvas, cv2.COLOR_GRAY2BGR)
            
            
        