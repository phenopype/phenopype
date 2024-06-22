#%% modules

import copy
import cv2
import numpy as np
import math
import sys
from dataclasses import make_dataclass

from phenopype import _vars, config
from phenopype import utils_lowlevel as ul


#%% settings

inf = math.inf

#%% functions

def draw_comment(
    image,
    annotations,
    label_colour="default",
    label_size="auto",
    label_width="auto",
    background=True,
    background_colour="white",
    background_pad=10,
    background_border="black",
    font="simplex",
    **kwargs,
):   
    """

    Parameters
    ----------
    image : ndarray
        image used as canvas 
    annotation: dict
        phenopype annotation containing QR-code (comment)
    line_colour: {"default", ... see phenopype.print_colours()} str, optional
        contour line colour - default colour as specified in settings
    line_width: {"auto", ... int > 0} int, optional 
        contour line width - automatically scaled to image by default
    label : bool, optional
        draw reference label
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        contour label colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        contour label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        contour label font thickness - automatically scaled to image by default
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    image: ndarray
        canvas with contours

    """
    # =============================================================================
    # setup
    
        
    label_size = ul._get_size(image.shape[1], image.shape[0], "label_size", label_size)
    label_width = ul._get_size(image.shape[1], image.shape[0], "label_width", label_width)
    
    label_colour = ul._get_bgr(label_colour, "label_colour")

    font = _vars.opencv_font_flags[font]

    if background:
        background_colour = ul._get_bgr(background_colour)
        background_border = ul._get_bgr(background_border)

    # =============================================================================
    # annotation management

    annotation_type = _vars._comment_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    # =============================================================================
    # execute
    
    canvas = copy.deepcopy(image)
    
    ## prep label
    name = annotation["data"]["label"]
    label_text = annotation["data"][_vars._comment_type]
    label =  name + ": " + label_text
    label_coords = (int(canvas.shape[0] // 10), int(canvas.shape[1] / 3))
            
    ## add background
    if background:
        
        text_size, _ = cv2.getTextSize(label, font, label_size, label_width)
        text_size = int(text_size[0] + background_pad), int(text_size[1] + (text_size[1]/8) + background_pad)
                
        background_coords = label_coords
        background_coords = int(background_coords[0] - (background_pad)), \
            int(background_coords[1] + (text_size[1]/8) + (background_pad))

        background_border_line_width = ul._get_size(image.shape[1], image.shape[0], "line_width")

        cv2.rectangle(
            canvas, 
            background_coords, 
            (label_coords[0] + text_size[0], label_coords[1] - text_size[1]), 
            background_colour, 
            -1
        )
        
        cv2.rectangle(
            canvas, 
            background_coords, 
            (label_coords[0] + text_size[0], label_coords[1] - text_size[1]), 
            background_border, 
            background_border_line_width
        )
        
    ## draw label
    cv2.putText(
        canvas,
        label,
        label_coords,
        font,
        label_size,
        label_colour,
        label_width,
        cv2.LINE_AA,
    )

    # =============================================================================
    # return

    return canvas

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
    bbox=False,
    bbox_ext=20,
    bbox_colour="default",
    bbox_line_width="auto",
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
    line_colour: {"default", ... see phenopype.print_colours()} str, optional
        contour line colour - default colour as specified in settings
    line_width: {"auto", ... int > 0} int, optional 
        contour line width - automatically scaled to image by default
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        contour label colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        contour label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        contour label font thickness - automatically scaled to image by default
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

    annotation_type = _vars._contour_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    contours = annotation["data"][annotation_type]
    contours_support = annotation["data"]["support"]
    
    if "contour_idx" in kwargs:
        contour_idx = kwargs.get("contour_idx")
        if contour_idx.__class__.__name__ == "int":
            contour_idx = [contour_idx]
        contours = [contours[i-1] for i in contour_idx]
        contours_support = [contours_support[i-1] for i in contour_idx]

    # =============================================================================
    # setup

    level = kwargs.get("level", 3)
    fill_colour = kwargs.get("fill_colour", line_colour)

    ## flags
    flags = make_dataclass(
        cls_name="flags",
        fields=[
            ("bbox", bool, bbox),
            ("label", bool, label),
            ("fill", bool, True),
        ],
    )

    line_width = ul._get_size(image.shape[1], image.shape[0], "line_width", line_width)
    label_size = ul._get_size(image.shape[1], image.shape[0], "label_size", label_size)
    label_width = ul._get_size(image.shape[1], image.shape[0], "label_width", label_width)
    bbox_line_width = ul._get_size(image.shape[1], image.shape[0], "line_width", bbox_line_width)

    fill_colour = ul._get_bgr(fill_colour, "line_colour")
    line_colour = ul._get_bgr(line_colour, "line_colour")
    label_colour = ul._get_bgr(label_colour, "label_colour")
    bbox_colour = ul._get_bgr(bbox_colour, "line_colour")

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
    if flags.bbox:
        q = bbox_ext
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            cv2.rectangle(
                canvas,
                (rx - q, ry - q),
                (rx + rw + q, ry + rh + q),
                bbox_colour,
                bbox_line_width,
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
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        contour label colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        contour label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        contour label font thickness - automatically scaled to image by default
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

    annotation_type = _vars._landmark_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = ul._get_annotation(
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
    
    point_size = ul._get_size(image.shape[1], image.shape[0], "point_size", point_size)
    label_size = ul._get_size(image.shape[1], image.shape[0], "label_size", label_size)
    label_width = ul._get_size(image.shape[1], image.shape[0], "label_width", label_width)

    point_colour = ul._get_bgr(point_colour, "point_colour")
    label_colour = ul._get_bgr(label_colour, "label_colour")
    
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
    fill=0,
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
    label : bool, optional
        draw mask label
    line_colour: {"default", ... see phenopype.print_colours()} str, optional
        contour line colour - default colour as specified in settings
    line_width: {"auto", ... int > 0} int, optional 
        contour line width - automatically scaled to image by default
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        contour label colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        contour label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        contour label font thickness - automatically scaled to image by default

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

    annotation_type = _vars._mask_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    polygons = annotation["data"][annotation_type]
    label = annotation["data"]["label"]

    # =============================================================================
    # setup

    line_width = ul._get_size(image.shape[1], image.shape[0], "line_width", line_width)
    label_size = ul._get_size(image.shape[1], image.shape[0], "label_size", label_size)
    label_width = ul._get_size(image.shape[1], image.shape[0], "label_width", label_width)
    
    line_colour = ul._get_bgr(line_colour, "line_colour")
    label_colour = ul._get_bgr(label_colour, "label_colour")

    # =============================================================================
    # execute

    canvas = copy.deepcopy(image)
    
    for coords in polygons:
        if fill > 0:
            cv2.fillPoly(canvas, np.array([coords]), line_colour)
        else:
            cv2.polylines(canvas, np.array([coords]), False, line_colour, line_width)
    
        if flags.label:
            if isinstance(coords[0], list):
                label_coords = tuple(coords[0])
            elif isinstance(coords[0], np.ndarray):
                label_coords = tuple(coords[0][0])
            elif isinstance(coords[0], tuple):
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
    
    if fill > 0:
        # Blend the original canvas with the filled polygons
        original_canvas = copy.deepcopy(image)
        canvas = cv2.addWeighted(original_canvas, 1 - fill, canvas, fill, 0)

    return canvas
        


def draw_polyline(
    image, 
    annotations, 
    line_colour="default", 
    line_width="auto", 
    show_nodes=False,
    node_colour="default",
    node_size="auto",   
    **kwargs
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
    line_colour: {"default", ... see phenopype.print_colours()} str, optional
        contour line colour - default colour as specified in settings
    line_width: {"auto", ... int > 0} int, optional 
        contour line width - automatically scaled to image by default
        DESCRIPTION. The default is "auto".
    show_nodes : bool, optional
        show nodes of polyline. The default is False.
    node_colour : str, optional
        colour of node points. The default is "default".
    node_size : int, optional
        size of node points. The default is "auto".

    Returns
    -------
    image: ndarray
        canvas with lines
    """

    
    # =============================================================================
    # setup

    show_nodes = kwargs.get("show_nodes", False)
    node_colour = kwargs.get("node_colour", "default")
    node_size = kwargs.get("node_size", "auto")

    ## flags
    flags = make_dataclass(cls_name="flags", 
                           fields=[("show_nodes", bool, show_nodes)])

    # =============================================================================
    # annotation management

    annotation_type = _vars._line_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    lines = annotation["data"][annotation_type]

    # =============================================================================
    # setup

    
    line_width = ul._get_size(image.shape[1], image.shape[0], "line_width", line_width)
    line_colour = ul._get_bgr(line_colour, "line_colour")
    if flags.show_nodes:
        node_size = ul._get_size(image.shape[1], image.shape[0], "point_size", node_size)
        node_colour = ul._get_bgr(node_colour, "line_colour")


    # =============================================================================
    # execute

    canvas = copy.deepcopy(image)

    ## draw lines
    for coords in lines:
        cv2.polylines(
            canvas, 
            np.array([coords]), 
            False, 
            line_colour, 
            line_width
            )
        if flags.show_nodes:
            for node in coords:
                print(node)
                cv2.circle(
                    canvas,
                    tuple(node),
                    node_size,
                    node_colour,
                    -1
                    )
                   
                   
    # =============================================================================
    # return

    return canvas


def draw_QRcode(
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
    image : ndarray
        image used as canvas 
    annotation: dict
        phenopype annotation containing QR-code (comment)
    line_colour: {"default", ... see phenopype.print_colours()} str, optional
        contour line colour - default colour as specified in settings
    line_width: {"auto", ... int > 0} int, optional 
        contour line width - automatically scaled to image by default
    label : bool, optional
        draw reference label
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        contour label colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        contour label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        contour label font thickness - automatically scaled to image by default
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    canvas : TYPE
        DESCRIPTION.

    """
    # =============================================================================
    # setup

    flags = make_dataclass(cls_name="flags", fields=[("label", bool, label)])
        
    line_width = ul._get_size(image.shape[1], image.shape[0], "line_width", line_width)
    label_size = ul._get_size(image.shape[1], image.shape[0], "label_size", label_size)
    label_width = ul._get_size(image.shape[1], image.shape[0], "label_width", label_width)
    
    line_colour = ul._get_bgr(line_colour, "line_colour")
    label_colour = ul._get_bgr(label_colour, "label_colour")

    # =============================================================================
    # annotation management

    annotation_type = _vars._comment_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
        
    points = annotation["data"][_vars._mask_type]
    label = annotation["data"][annotation_type]

    canvas = cv2.polylines(
        image, 
        [np.asarray(points, np.int32)], 
        True, 
        line_colour, 
        line_width)

    if flags.label:
        
        (x,y), _ = cv2.minEnclosingCircle(np.array(points))

        label_coords = (int(x),int(y))

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
    image : ndarray
        image used as canvas 
    annotation: dict
        phenopype annotation containing reference data
    line_colour: {"default", ... see phenopype.print_colours()} str, optional
        contour line colour - default colour as specified in settings
    line_width: {"auto", ... int > 0} int, optional 
        contour line width - automatically scaled to image by default
    label : bool, optional
        draw reference label
    label_colour : {"default", ... see phenopype.print_colours()} str, optional
        contour label colour - default colour as specified in settings
    label_size: {"auto", ... int > 0} int, optional 
        contour label font size - automatically scaled to image by default
    label_width:  {"auto", ... int > 0} int, optional 
        contour label font thickness - automatically scaled to image by default

    Returns
    -------
    canvas : TYPE
        DESCRIPTION.

    """

    # =============================================================================
    # annotation management

    annotation_type = _vars._reference_type
    annotation_id = kwargs.get(annotation_type + "_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

    px_ratio, unit = annotation["data"][annotation_type]
    polygons = annotation["data"][_vars._mask_type]

    # =============================================================================
    # setup

    ## flags
    flags = make_dataclass(cls_name="flags", fields=[("label", bool, label)])
    
    line_width = ul._get_size(image.shape[1], image.shape[0], "line_width", line_width)
    label_size = ul._get_size(image.shape[1], image.shape[0], "label_size", label_size)
    label_width = ul._get_size(image.shape[1], image.shape[0], "label_width", label_width)
    
    line_colour = ul._get_bgr(line_colour, "line_colour")
    label_colour = ul._get_bgr(label_colour, "label_colour")

    # =============================================================================
    # execute

    canvas = copy.deepcopy(image)

    ## draw referenc mask outline
    if len(annotation["data"][_vars._mask_type]) > 0:
        mask_coord_list = annotation["data"][_vars._mask_type]
        cv2.polylines(canvas, np.array([mask_coord_list[0]]), False, line_colour, line_width)
    
    if "support" in annotation["data"]:
        point_coord_list = annotation["data"]["support"]
        cv2.polylines(canvas, np.array([point_coord_list]), False, line_colour, line_width)


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
            canvas, np.array([scale_box]), ul._get_bgr("lightgrey")
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
            line_colour,
            line_width,
        )

        cv2.putText(
            canvas,
            "10 " + unit,
            (int(wp * 6), int(wp * 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_size,
            label_colour,
            label_width * 2,
            cv2.LINE_AA,
        )

    # =============================================================================
    # return

    return canvas


def select_canvas(image, canvas="raw", multi_channel=True, invert=False, **kwargs):
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

    if image.__class__.__name__ == "_Container":
        
        ## method
        if canvas == "mod":
            image.canvas = copy.deepcopy(image.image)
            ul._print("- modifed image")
        elif canvas == "raw":
            image.canvas = copy.deepcopy(image.image_copy)
            ul._print("- raw image")
        elif canvas == "bin":
            image.canvas = copy.deepcopy(image.image_bin)
            ul._print("- binary image")
        elif canvas == "gray":
            image.canvas = cv2.cvtColor(image.image_copy, cv2.COLOR_BGR2GRAY)
            ul._print("- grayscale image")
        elif canvas == "blue":
            image.canvas = image.image_copy[:, :, 0]
            ul._print("- blue channel")
        elif canvas == "green":
            image.canvas = image.image_copy[:, :, 1]
            ul._print("- green channel")
        elif canvas == "red":
            image.canvas = image.image_copy[:, :, 2]
            ul._print("- red channel")
        else:
            ul._print("- invalid selection - defaulting to raw image", lvl=2)
            image.canvas = copy.deepcopy(image.image_copy)
            
        if invert == True:
            image.canvas = cv2.bitwise_not(image.canvas)
            ul._print("- inverted image")


    elif image.__class__.__name__ == "ndarray":
        if canvas == "raw":
            canvas = copy.deepcopy(image)
            ul._print("- raw image")
        elif canvas == "gray":
            canvas = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ul._print("- grayscale image")
        elif canvas == "blue":
            canvas = image[:, :, 0]
            ul._print("- blue channel")
        elif canvas == "green":
            canvas = image[:, :, 1]
            ul._print("- green channel")
        elif canvas == "red":
            canvas = image[:, :, 2]
            ul._print("- red channel")  
        else:
            canvas = copy.deepcopy(image)
            ul._print("- invalid selection - defaulting to raw image", lvl=2)
            
        if invert == True:
            canvas = cv2.bitwise_not(canvas)
            ul._print("- inverted image")

    ## check if colour
    if multi_channel:
        if image.__class__.__name__ == "_Container":
            if len(image.canvas.shape) < 3:
                image.canvas = cv2.cvtColor(image.canvas, cv2.COLOR_GRAY2BGR)
        elif image.__class__.__name__ == "ndarray":
            if len(canvas.shape) < 3:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
                
    return canvas
