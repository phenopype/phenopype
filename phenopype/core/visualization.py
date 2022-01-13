#%% modules

import copy
import cv2
import numpy as np
import math
from dataclasses import make_dataclass

from phenopype import settings
from phenopype import utils_lowlevel


#%% settings

inf = math.inf

#%% functions


def draw_contour(
    image,
    annotations,
    fill=0.3,
    line_colour="default",
    line_width="auto",
    label=False,
    label_colour="default",
    label_size="auto",
    label_width="auto",
    offset_coords=None,
    bounding_box=False,
    bounding_box_ext=20,
    bounding_box_colour="default",
    bounding_box_line_width="auto",
    **kwargs,
):
    """
    Draw contours and their labels onto a canvas. Can be filled or empty, offset
    coordinates can be supplied. 

    Parameters
    ----------
    image : ndarray
        image used as canvas 
    annotation: dict
        phenopype annotation containing contours
    offset_coords : tuple, optional
        offset coordinates, will be added to all contours
    label : bool, optional
        draw contour label
    fill : float, optional
        background transparency for contour fill (0=no fill).
    level : int, optional
        the default is 3.
    line_colour: {"green", "red", "blue", "black", "white"} str, optional
        contour line colour
    line_width: int, optional
        contour line width
    label_colour : {"black", "white", "green", "red", "blue"} str, optional
        contour label colour.
    label_font_size: int, optional
        contour label font size (scaled to image)
    label_font_width: int, optional
        contour label font thickness 
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
    image: ndarray
        canvas with contours

    """

    # =============================================================================
    # annotation management

    annotation_type = settings._contour_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    contours = annotation["data"][annotation_type]
    contours_support = annotation["data"]["support"]

    # =============================================================================
    # setup

    level = kwargs.get("level", 3)
    fill_colour = kwargs.get("fill_colour", line_colour)

    ## flags
    flags = make_dataclass(
        cls_name="flags",
        fields=[
            ("bounding_box", bool, bounding_box),
            ("label", bool, label),
            ("fill", bool, True),
        ],
    )

    if line_width == "auto":
        line_width = utils_lowlevel._auto_line_width(image, factor=0.001)
    if label_size == "auto":
        label_size = utils_lowlevel._auto_text_size(image)
    if label_width == "auto":
        label_width = utils_lowlevel._auto_text_width(image)
    if bounding_box_line_width == "auto":
        bounding_box_line_width = utils_lowlevel._auto_line_width(image)

    if fill_colour == "default":
        fill_colour = settings._default_line_colour
    if line_colour == "default":
        line_colour = settings._default_line_colour
    if label_colour == "default":
        label_colour = settings._default_label_colour
    if bounding_box_colour == "default":
        bounding_box_colour = settings._default_line_colour

    fill_colour = utils_lowlevel._get_bgr(fill_colour)
    line_colour = utils_lowlevel._get_bgr(line_colour)
    label_colour = utils_lowlevel._get_bgr(label_colour)
    bounding_box_colour = utils_lowlevel._get_bgr(bounding_box_colour)

    ## filling and line settings
    if fill > 0:
        flags.fill = True
    else:
        flags.fill = False

    # =============================================================================
    # execute

    canvas = copy.deepcopy(image)

    ## 1) fill contours
    if flags.fill:
        colour_mask = copy.deepcopy(canvas)
        for contour in contours:
            cv2.drawContours(
                image=canvas,
                contours=[contour],
                contourIdx=0,
                thickness=-1,
                color=line_colour,
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
                bounding_box_colour,
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
    annotations,
    label=True,
    label_colour="default",
    label_size="auto",
    label_width="auto",
    offset=0,
    point_colour="default",
    point_size="auto",
    **kwargs,
):
    """
    Draw landmarks into an image.

    Parameters
    ----------
    image : ndarray
        image used as canvas 
    annotation: dict
        phenopype annotation containing landmarks
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
    image: ndarray
        canvas with landmarks

    """

    # =============================================================================
    # annotation management

    annotation_type = settings._landmark_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    points = annotation["data"][annotation_type]

    # =============================================================================
    # setup

    ## flags
    flags = make_dataclass(cls_name="flags", fields=[("label", bool, label)])

    ## configure points
    if point_size == "auto":
        point_size = utils_lowlevel._auto_point_size(image)
    if label_size == "auto":
        label_size = utils_lowlevel._auto_text_size(image)
    if label_width == "auto":
        label_width = utils_lowlevel._auto_text_width(image)

    if label_colour == "default":
        label_colour = settings._default_label_colour
    if point_colour == "default":
        point_colour = settings._default_point_colour

    label_colour = utils_lowlevel._get_bgr(label_colour)
    point_colour = utils_lowlevel._get_bgr(point_colour)

    # =============================================================================
    # execute

    canvas = copy.deepcopy(image)

    for idx, point in enumerate(points):
        x, y = point
        cv2.circle(canvas, (x, y), point_size, point_colour, -1)
        if flags.label:
            x, y = x + offset, y + offset
            cv2.putText(
                canvas,
                str(idx + 1),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                label_colour,
                label_width,
                cv2.LINE_AA,
            )

    # =============================================================================
    # return

    return canvas


def draw_mask(
    image,
    annotations,
    line_colour="default",
    line_width="auto",
    label=False,
    label_colour="default",
    label_size="auto",
    label_width="auto",
    **kwargs,
):
    """
    Draw masks into an image. This function is also used to draw the perimeter 
    of a created or detected reference card.
    
    Parameters
    ----------        
    image : ndarray
        image used as canvas 
    annotation: dict
        phenopype annotation containing masks
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
    image: ndarray
        canvas with masks
    """

    # =============================================================================
    # setup

    ## flags
    flags = make_dataclass(cls_name="flags", fields=[("label", bool, label)])

    # =============================================================================
    # annotation management

    annotation_type = settings._mask_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    polygons = annotation["data"][annotation_type]
    label = annotation["data"]["label"]

    # =============================================================================
    # setup

    if line_width == "auto":
        line_width = utils_lowlevel._auto_line_width(image)
    if label_size == "auto":
        label_size = utils_lowlevel._auto_text_size(image)
    if label_width == "auto":
        label_width = utils_lowlevel._auto_text_width(image)

    if line_colour == "default":
        line_colour = settings._default_line_colour
    if label_colour == "default":
        label_colour = settings._default_label_colour

    label_colour = utils_lowlevel._get_bgr(label_colour)
    line_colour = utils_lowlevel._get_bgr(line_colour)

    # =============================================================================
    # execute

    canvas = copy.deepcopy(image)

    for coords in polygons:
        cv2.polylines(
            canvas, np.array([coords]), False, line_colour, line_width,
        )

        if flags.label:

            if coords[0].__class__.__name__ == "list":
                label_coords = tuple(coords[0])
            elif coords[0].__class__.__name__ == "ndarray":
                label_coords = tuple(coords[0][0])
            elif coords[0].__class__.__name__ == "tuple":
                label_coords = coords[0]

            cv2.putText(
                canvas,
                label,
                label_coords,
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
    image, annotations, line_colour="default", line_width="auto", **kwargs
):
    """
    Draw masks into an image. This function is also used to draw the perimeter 
    of a created or detected reference card.
    
    Parameters
    ----------        
    image : ndarray
        image used as canvas 
    annotation: dict
        phenopype annotation containing lines
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
    image: ndarray
        canvas with lines
    """

    # =============================================================================
    # annotation management

    annotation_type = settings._line_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    lines = annotation["data"][annotation_type]

    # =============================================================================
    # setup

    ## line settings
    if line_width == "auto":
        line_width = utils_lowlevel._auto_line_width(image)

    if line_colour == "default":
        line_colour = settings._default_line_colour

    line_colour = utils_lowlevel._get_bgr(line_colour)

    # =============================================================================
    # execute

    canvas = copy.deepcopy(image)

    ## draw lines
    for coords in lines:
        cv2.polylines(canvas, np.array([coords]), False, line_colour, line_width)

    # =============================================================================
    # return

    return canvas


def draw_reference(
    image,
    annotations,
    line_colour="default",
    line_width="auto",
    label=True,
    label_colour="default",
    label_size="auto",
    label_width="auto",
    **kwargs,
):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    annotations : TYPE
        DESCRIPTION.
    line_colour : TYPE, optional
        DESCRIPTION. The default is "blue".
    line_width : TYPE, optional
        DESCRIPTION. The default is "auto".
    scale : TYPE, optional
        DESCRIPTION. The default is False.
    label_size : TYPE, optional
        DESCRIPTION. The default is "auto".
    label_colour : TYPE, optional
        DESCRIPTION. The default is "blue".
    label_width : TYPE, optional
        DESCRIPTION. The default is "auto".

    Returns
    -------
    canvas : TYPE
        DESCRIPTION.

    """

    # =============================================================================
    # annotation management

    annotation_type = settings._reference_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = utils_lowlevel._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    px_ratio, unit = annotation["data"][annotation_type]
    polygons = annotation["data"][settings._mask_type]

    # =============================================================================
    # setup

    ## flags
    flags = make_dataclass(cls_name="flags", fields=[("label", bool, label)])

    if line_width == "auto":
        line_width = utils_lowlevel._auto_line_width(image)
    if label_size == "auto":
        label_size = utils_lowlevel._auto_text_size(image)
    if label_width == "auto":
        label_width = utils_lowlevel._auto_text_width(image)
    if line_colour == "default":
        line_colour = settings._default_line_colour
    if label_colour == "default":
        label_colour = settings._default_label_colour

    label_colour = utils_lowlevel._get_bgr(label_colour)
    line_colour = utils_lowlevel._get_bgr(line_colour)

    # =============================================================================
    # execute

    canvas = copy.deepcopy(image)

    ## draw referenc mask outline
    cv2.polylines(canvas, np.array([polygons[0]]), False, line_colour, line_width)

    ## draw scale
    if flags.label:
        height, width = canvas.shape[:2]

        hp, wp = height / 100, width / 100

        length = int(px_ratio * 10)

        scale_box = [
            (int(wp * 3), int(hp * 3)),
            (int((wp * 3) + length + (wp * 4)), int(hp * 3)),
            (int((wp * 3) + length + (wp * 4)), int(hp * 11)),
            (int(wp * 3), int(hp * 11)),
            (int(wp * 3), int(hp * 3)),
        ]

        cv2.fillPoly(
            canvas, np.array([scale_box]), utils_lowlevel._get_bgr("lightgrey")
        )

        scale_box_inner = [
            (int(wp * 5), int(hp * 5)),
            (int((wp * 5) + length), int(hp * 5)),
            (int((wp * 5) + length), int(hp * 9)),
            (int(wp * 5), int(hp * 9)),
            (int(wp * 5), int(hp * 5)),
        ]

        cv2.fillPoly(canvas, np.array([scale_box_inner]), line_colour)

        cv2.polylines(
            canvas,
            np.array([scale_box_inner]),
            False,
            utils_lowlevel._get_bgr("black"),
            utils_lowlevel._auto_line_width(canvas, factor=0.001),
        )

        cv2.putText(
            canvas,
            "10 " + unit,
            (int(wp * 6), int(wp * 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            utils_lowlevel._auto_text_size(canvas),
            utils_lowlevel._get_bgr("black"),
            label_width * 2,
            cv2.LINE_AA,
        )

    # =============================================================================
    # return

    return canvas


def select_canvas(image, canvas="raw", multi_channel=True, **kwargs):
    """
    Isolate a colour channel from an image or select canvas for the pype method.

    Parameters
    ----------
    image : ndarray
        image used as canvas 
    canvas : {"mod", "bin", "gray", "raw", "red", "green", "blue"} str, optional
        the type of canvas to be used for visual feedback. some types require a
        function to be run first, e.g. "bin" needs a segmentation algorithm to be
        run first. black/white images don't have colour channels. coerced to 3D
        array by default
    multi: bool, optional
        coerce returned array to multichannel (3-channel)

    Returns
    -------
    canvas : ndarray
        canvas for drawing

    """

    if image.__class__.__name__ == "Container":

        ## method
        if canvas == "mod":
            image.canvas = copy.deepcopy(image.image)
            print("- modifed image")
        elif canvas == "raw":
            image.canvas = copy.deepcopy(image.image_copy)
            print("- raw image")
        # elif canvas == "bin":
        #     image.canvas = copy.deepcopy(image.image_bin)
        # print("- binary image")
        # elif canvas == "gray":
        #     image.canvas = cv2.cvtColor(image.image_gray, cv2.COLOR_BGR2GRAY)
        #     print("- grayscale image")
        elif canvas == "green":
            image.canvas = image.image_copy[:, :, 0]
            print("- green channel")
        elif canvas == "red":
            image.canvas = image.image_copy[:, :, 1]
            print("- red channel")
        elif canvas == "blue":
            image.canvas = image.image_copy[:, :, 2]
            print("- blue channel")
        else:
            print("- invalid selection - defaulting to raw image")
            image.canvas = copy.deepcopy(image.image_copy)

    elif image.__class__.__name__ == "ndarray":
        if canvas == "raw":
            canvas = copy.deepcopy(image)
            print("- raw image")
        elif canvas == "gray":
            canvas = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("- grayscale image")
        elif canvas == "green":
            canvas = image[:, :, 0]
            print("- green channel")
        elif canvas == "red":
            canvas = image[:, :, 1]
            print("- red channel")
        elif canvas == "blue":
            canvas = image[:, :, 2]
            print("- blue channel")
        else:
            canvas = copy.deepcopy(image)
            print("- invalid selection - defaulting to raw image")

    ## check if colour
    if multi_channel:
        if image.__class__.__name__ == "Container":
            if len(image.canvas.shape) < 3:
                image.canvas = cv2.cvtColor(image.canvas, cv2.COLOR_GRAY2BGR)
        elif image.__class__.__name__ == "ndarray":
            if len(canvas.shape) < 3:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    return canvas
