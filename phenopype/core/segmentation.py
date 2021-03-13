#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from math import inf

from phenopype.settings import colours
from phenopype.core.preprocessing import invert_image
from phenopype.utils_lowlevel import _create_mask_bool, _image_viewer, _auto_line_width
from phenopype.core.visualization import draw_contours

#%% functions


def blur(obj_input, kernel_size=5, method="averaging", sigma_color=75, sigma_space=75):
    """
    Apply a blurring algorithm to the image.

    Parameters
    ----------
    obj_input : array or container
        input object
    kernel_size: int, optional
        size of the blurring kernel (has to be odd - even numbers will be ceiled)
    method: {averaging, gaussian, median, bilateral} str, optional
        blurring algorithm
    sigma_colour: int, optional
        for 'bilateral'
    sigma_space: int, optional
        for 'bilateral'

    Returns
    -------
    image : array or container
        blurred image

    """
    ## kwargs
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image)

    ## method
    if method == "averaging":
        image = cv2.blur(image, (kernel_size, kernel_size))
    elif method == "gaussian":
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        image = cv2.medianBlur(image, kernel_size)
    elif method == "bilateral":
        image = cv2.bilateralFilter(image, kernel_size, sigma_color, sigma_space)
    else:
        image = image

    ## return
    if obj_input.__class__.__name__ == "container":
        obj_input.image = image
    else:
        return image



def draw(
    obj_input,
    df_image_data=None,
    mode="silent",
    overwrite=False,
    edit=False,
    label="drawing1",
    tool="line",
    canvas="image",
    line_width="auto",
    line_colour="black",
    max_dim=None,
    **kwargs
):
    """
    Set points, draw a connected line between them, and measure its length. 

    Parameters
    ----------
    obj_input : array or container
        input object
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata, will be added to
        output DataFrame
    overwrite: bool, optional
        if working using a container, or from a phenopype project directory, 
        should existing polylines be overwritten
    label: str, optinal
        assigns a label to the drawing
    tool : {line, polygon, rectangle} str, optional
        type of tool to use for drawing
    line_colour : {"black", "white"} str, optional
        line or filling (for rectangle and polygon) colour
    line_width : int, optional
        line width

    Returns
    -------
    df_polylines : DataFrame or container
        contains the drawn polylines

    """
    ## kwargs
    flag_overwrite = overwrite
    flag_edit = edit
    flag_canvas = canvas
    test_params = kwargs.get("test_params", None)
    
    prev_drawings = {}
    
    ## load image
    df_drawings = None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename": "unknown"}, index=[0])
    elif obj_input.__class__.__name__ == "container":
        if flag_canvas == "image":
            image = copy.deepcopy(obj_input.image)
        elif flag_canvas == "canvas":
            image = copy.deepcopy(obj_input.canvas)
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_drawings"):
            df_drawings = obj_input.df_drawings
            if "index" in df_drawings:
                df_drawings.drop(columns = ["index"], inplace=True)
            df_drawings.reset_index(drop=True, inplace=True)
        if hasattr(obj_input, "image_bin"):
            image_bin = obj_input.image_bin
        else:
            print("no binary image provided")
            return
    
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if line_width == "auto":
        line_width = _auto_line_width(image)
    test_params = kwargs.get("test_params", None)

    while True:
        
        if not df_drawings.__class__.__name__ == "NoneType":
            ## select df_drawing that matches label
            df_drawings_sub = df_drawings.loc[df_drawings['label'] == label]
            ## remove meta-columns
            df_drawings_sub = df_drawings_sub[
                df_drawings_sub.columns.intersection(["label", "tool", "line_colour","line_width","coords"])
            ]
        
        ## check if exists
        if not df_drawings.__class__.__name__ == "NoneType" and flag_overwrite == False and flag_edit == False:
            if label in df_drawings_sub["label"].values:
                print("- drawing with label " + label + " already created (overwrite=False)")
                break
        elif not df_drawings.__class__.__name__ == "NoneType" and flag_edit == True:
            ## extract previous drawing and convert to dict
            if label in df_drawings_sub["label"].values:
                prev_point_list = []
                for index, row in df_drawings_sub.iterrows():
                    coords = eval(row["coords"])
                    prev_point_list.append(coords) 
                prev_drawings = {"tool": df_drawings_sub["tool"].unique()[0],
                                 "line_colour": colours[df_drawings_sub["line_colour"].unique()[0]],
                                 "line_width": df_drawings_sub["line_width"].unique()[0],
                                 "point_list": prev_point_list}
                
                ## remove rows from original drawing df
                df_drawings = df_drawings.drop(df_drawings[df_drawings["label"] == label].index)
                print("- drawing (editing)")
            pass
        elif not df_drawings.__class__.__name__ == "NoneType" and flag_overwrite == True:
            if label in df_drawings["label"].values:            
                ## remove rows from original drawing df
                df_drawings = df_drawings.drop(df_drawings[df_drawings["label"] == label].index)
                print("- drawing (overwriting)")
            pass
        
        elif df_drawings.__class__.__name__ == "NoneType" or len(df_drawings) == 0:
            df_drawings = pd.DataFrame(columns=["label", "tool", "line_colour","line_width","coords"])
            prev_drawings = {}
            print("- drawing")
            pass
        elif mode == "silent":
            print("Silent mode - using existing coordinates")
            break

        ## add other labels for reference
        if len(df_drawings) > 0:
            for idx, row in df_drawings.iterrows():
                coords = eval(row["coords"])
                if row["tool"] in ["line", "lines","polyline","polylines"]:
                    cv2.polylines(
                        image,
                        np.array([coords]),
                        False,
                        colours[row["line_colour"]],
                        row["line_width"],
                    )
                elif row["tool"] in ["rect", "rectangle", "poly", "polygon"]:
                    cv2.fillPoly(image, np.array([coords]), colours[row["colour"]])

        

        ## method
        if not test_params.__class__.__name__ == "NoneType":
            out = _image_viewer(image,tool=tool, draw=True, 
                    line_width=line_width,line_colour=line_colour,
                    previous=test_params,max_dim = max_dim)
        elif not df_drawings.__class__.__name__ == "NoneType" and flag_edit == True:
            out = _image_viewer(image,tool=tool, draw=True, 
                                line_width=line_width,line_colour=line_colour,
                                previous=prev_drawings,max_dim = max_dim)
        else:
            out = _image_viewer(image,tool=tool, draw=True, 
                                line_width=line_width,line_colour=line_colour,
                                max_dim = max_dim)
        coords = out.point_list
        
        ## abort
        if not out.done:
            if obj_input.__class__.__name__ == "ndarray":
                print("terminated drawing")
                return
            elif obj_input.__class__.__name__ == "container":
                print("- terminated drawing")
                return True

        ## create df
        df_drawings_sub_new = pd.DataFrame()
        if len(coords) > 0:
            for points in coords:
                df_drawings_sub_new = df_drawings_sub_new.append(
                    {"label": label, 
                     "tool": tool, 
                     "line_colour": line_colour, 
                     "line_width": line_width,
                     "coords": str(points)},
                    ignore_index=True,
                    sort=True,
                )
            df_drawings_sub_new.line_width = df_drawings_sub_new.line_width.astype(int)
        else:
            print("zero coordinates - redo drawing!")
            
        ## merge with existing image_data frame
        df_drawings_sub_new = pd.concat(
            [
                pd.concat([df_image_data] * len(df_drawings_sub_new)).reset_index(drop=True),
                df_drawings_sub_new.reset_index(drop=True),
            ],
            axis=1,
        )
        df_drawings = df_drawings.append(df_drawings_sub_new)
        break
    
    ## draw
    for idx, row in df_drawings.loc[df_drawings['label'] == label].iterrows():
        coords = eval(row["coords"])
        if row["tool"] in ["line", "lines","polyline","polylines"]:
            cv2.polylines(
                image_bin,
                np.array([coords]),
                False,
                colours[row["line_colour"]],
                row["line_width"],
            )
        elif row["tool"] in ["rect", "rectangle", "poly", "polygon"]:
            cv2.fillPoly(image_bin, np.array([coords]), colours[row["colour"]])


    ## drop index before saving
    df_drawings.reset_index(drop=True, inplace=True)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image_bin, df_drawings
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_drawings = df_drawings
        obj_input.image = image_bin

# pp.show_image(image)

def find_contours(
    obj_input,
    df_image_data=None,
    approximation="simple",
    retrieval="ext",
    offset_coords=(0, 0),
    min_nodes=3,
    max_nodes=inf,
    min_area=0,
    max_area=inf,
    min_diameter=0,
    max_diameter=inf,
    subset=None,
):
    """
    Find objects in binarized image. The input image needs to be a binarized image
    or a container on which a binarizing segmentation algorithm (e.g. threshold or 
    watershed) has been performed. A flavor of different approximation algorithms 
    and retrieval intstructions can be applied. The contours can also be filtered
    for minimum area, diameter or nodes (= number of corners). 

    Parameters
    ----------
    obj_input : array or container
        input object (binary)
    df_image_data : DataFrame, optional
        an existing DataFrame containing image metadata 
    approximation : {"none", "simple", "L1", "KCOS"] str, optional
        contour approximation algorithm:
            - none: no approximation, all contour coordinates are returned
            - simple: only the minimum coordinates required are returned
            - L1: Teh-Chin chain approximation algorithm 
            - KCOS: Teh-Chin chain approximation algorithm 
    retrieval : {"ext", "list", "tree", "ccomp", "flood"} str, optional
        contour retrieval procedure:
            - ext: retrieves only the extreme outer contours
            - list: retrieves all of the contours without establishing any 
              hierarchical relationships
            - tree: retrieves all of the contours and reconstructs a full 
              hierarchy of nested contours
            - ccomp: retrieves all of the contours and organizes them into a 
              two-level hierarchy (parent and child)
            - flood: flood-fill algorithm
    offset_coords : tuple, optional
        offset by which every contour point is shifted.
    subset: {"parent", "child"} str, optional
        retain only a subset of the parent-child order structure
    min_nodes : int, optional
        minimum number of coordinates
    max_nodes : int, optional
        maximum number of coordinates 
    min_area : int, optional
        minimum contour area in pixels
    max_area : int, optional
        maximum contour area in pixels
    min_diameter : int, optional
        minimum diameter of boundary circle
    max_diameter : int, optional
        maximum diameter of boundary circle

    Returns
    -------
    df_contours : DataFrame or container
        contains contours

    """

    ## kwargs
    retr_alg = {
        "ext": cv2.RETR_EXTERNAL,  ## only external
        "list": cv2.RETR_LIST,  ## all contours
        "tree": cv2.RETR_TREE,  ## fully hierarchy
        "ccomp": cv2.RETR_CCOMP,  ## outer perimeter and holes
        "flood": cv2.RETR_FLOODFILL,
    }  ## not sure what this does
    approx_alg = {
        "none": cv2.CHAIN_APPROX_NONE,  ## no approximation of the contours, all points
        "simple": cv2.CHAIN_APPROX_SIMPLE,  ## minimal corners
        "L1": cv2.CHAIN_APPROX_TC89_L1,  ## algorithm 1
        "KCOS": cv2.CHAIN_APPROX_TC89_KCOS,
    }  ## algorithm 2
    flag_subset = subset

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename": "unknown"}, index=[0])
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image_bin
        df_image_data = obj_input.df_image_data
    else:
        print("wrong input format.")
        return

    ## check
    if len(image.shape) > 2:
        print("Multi-channel array supplied - need binary array.")

    ## method
    image, contour_list, hierarchy = cv2.findContours(
        image=image,
        mode=retr_alg[retrieval],
        method=approx_alg[approximation],
        offset=offset_coords,
    )

    ## filtering
    if not contour_list.__class__.__name__ == "NoneType" and len(contour_list) > 0:
        contour_dict = {}
        idx = 0
        for contour, hier in zip(contour_list, hierarchy[0]):

            ## number of contour nodes
            if len(contour) > min_nodes and len(contour) < max_nodes:

                ## measure basic shape features
                center, radius = cv2.minEnclosingCircle(contour)
                center = int(center[0]), int(center[1])
                diameter = int(radius * 2)
                area = int(cv2.contourArea(contour))

                ## contour hierarchy
                if hier[3] == -1:
                    cont_order = "parent"
                else:
                    cont_order = "child"

                ## retain only a subset
                if not flag_subset.__class__.__name__ == "NoneType":
                    if flag_subset == cont_order:
                        pass
                    else:
                        continue

                ## feature filter
                if all(
                    [
                        diameter > min_diameter and diameter < max_diameter,
                        area > min_area and area < max_area,
                    ]
                ):
                    idx += 1
                    contour_label = str(idx)
                    contour_dict[contour_label] = {
                        "contour": contour_label,
                        "center": center,
                        "diameter": diameter,
                        "area": area,
                        "order": cont_order,
                        "idx_child": hier[2],
                        "idx_parent": hier[3],
                        "coords": contour,
                    }
        print("- found " + str(len(contour_dict)) + " contours that match criteria")
    else:
        print("- no contours found")

    ## output
    df_contours = pd.DataFrame(contour_dict).T
    df_contours.reset_index(drop=True, inplace=True)
    df_contours = pd.concat(
        [
            pd.concat([df_image_data] * len(df_contours)).reset_index(drop=True),
            df_contours.reset_index(drop=True),
        ],
        axis=1,
    )

    if obj_input.__class__.__name__ == "ndarray":
        return df_contours
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_contours = df_contours


def morphology(obj_input, kernel_size=5, shape="rect", operation="close", iterations=1):
    """
    Performs advanced morphological transformations using erosion and dilation 
    as basic operations. Provides different kernel shapes and a suite of operation
    types (read more about morphological operations here:
    https://docs.opencv.org/master/db/df6/tutorial_erosion_dilatation.html)

    Parameters
    ----------
    obj_input : array or container
        input object
    kernel_size: int, optional
        size of the morphology kernel (has to be odd - even numbers will be 
        ceiled)
    shape : {"rect", "cross", "ellipse"} str, optional
        shape of the kernel
    operation : {erode, dilate, open, close, gradient, tophat, blackhat, hitmiss} str, optional
        the morphology operation to be performed:
            - erode: remove pixels from the border 
            - dilate: add pixels to the border
            - open: erosion followed by dilation
            - close: dilation followed by erosion
            - gradient: difference between dilation and erosion of an input image
            - tophat: difference between input image and opening of input image
            - blackhat: difference between input image and closing of input image
            - hitmiss: find patterns in binary images (read more here:
              https://docs.opencv.org/master/db/d06/tutorial_hitOrMiss.html)
    iterations : int, optional
        number of times to run morphology operation

    Returns
    -------
    image : array or container
        processed image

    """
    ## kwargs
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    shape_list = {
        "cross": cv2.MORPH_CROSS,
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    operation_list = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat ": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
        "hitmiss": cv2.MORPH_HITMISS,
    }
    kernel = cv2.getStructuringElement(shape_list[shape], (kernel_size, kernel_size))

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image

    ## method
    image = cv2.morphologyEx(
        image, op=operation_list[operation], kernel=kernel, iterations=iterations
    )

    ## return
    if obj_input.__class__.__name__ == "container":
        obj_input.image = image
        obj_input.image_bin = image
    else:
        return image


def threshold(
    obj_input,
    df_masks=None,
    method="otsu",
    constant=1,
    blocksize=99,
    value=127,
    channel="gray",
    invert=False,
):
    """
    Applies a fixed-level threshold to create a binary image from a grayscale 
    image or a multichannel image (gray, red, green or blue channel can be selected).
    Three types of thresholding algorithms are supported. Binary mask coordinates
    can be supplied to include or exclude areas.

    Parameters
    ----------
    obj_input : array or container
        input object
    df_masks : DataFrame, optional
        contains mask coordinates
    method : {"otsu", "adaptive", "binary"} str, optional
        type of thresholding algorithm:
            - otsu: use Otsu algorithm to choose the optimal threshold value
            - adaptive: dynamic threshold values across image (uses arguments
              "blocksize" and "constant")
            - binary: fixed threshold value (uses argument "value")
    blocksize: int, optional
        Size of a pixel neighborhood that is used to calculate a threshold 
        value for the pixel (has to be odd - even numbers will be ceiled; for
        "adaptive" method)
    constant : int, optional
        value to subtract from binarization output (for "adaptive" method)
    value : {between 0 and 255} int, optional
        thesholding value (for "binary" method)
    channel {"gray", "red", "green", "blue"}: str, optional
        which channel of the image to use for thresholding 
    invert : bool, optional
        invert all pixel values PRIOR to applying threshold values

    Returns
    -------
    image : array or container
        binary image

    """

    ##kwargs
    if blocksize % 2 == 0:
        blocksize = blocksize + 1
    if value > 255:
        value = 255
    elif value < 0:
        value = 0

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        image = copy.deepcopy(obj_input.image)
        if hasattr(obj_input, "df_masks"):
            df_masks = copy.deepcopy(obj_input.df_masks)

    ## channel
    if len(image.shape) == 3:
        if channel == "gray" or channel == "grayscale":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel == "g" or channel == "green":
            image = image[:, :, 0]
        elif channel == "r" or channel == "red":
            image = image[:, :, 1]
        elif channel == "blue" or channel == "b":
            image = image[:, :, 2]

    if invert:
        image = invert_image(image)

    ## method
    if method == "otsu":
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blocksize,
            constant,
        )
    elif method == "binary":
        ret, image = cv2.threshold(image, value, 255, cv2.THRESH_BINARY_INV)

    ## apply masks
    if not df_masks.__class__.__name__ == "NoneType":
        ## include == True
        mask_bool, include_idx, exclude_idx = np.zeros(image.shape, dtype=bool), 0,0
        for index, row in df_masks.iterrows():
            if row["include"] == True:
                if not row["mask"] == "":
                    coords = eval(row["coords"])
                    mask_bool = np.logical_or(mask_bool, _create_mask_bool(image, coords))
                    include_idx += 1
        if include_idx > 0:
            image[mask_bool == False] = 0
        for index, row in df_masks.iterrows():
            if row["include"] == False:
                if not row["mask"] == "":
                    coords = eval(row["coords"])
                    image[_create_mask_bool(image, coords)] = 0
                    exclude_idx += 1
        if exclude_idx>0:
            print("- excluding pixels from " + str(exclude_idx) + " drawn masks ")
        if include_idx>0:
            print("- including pixels from " + str(include_idx) + " drawn masks ")

        
    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.image = image
        obj_input.image_bin = image


def watershed(
    obj_input,
    image_thresh=None,
    iterations=1,
    kernel_size=3,
    distance_cutoff=0.8,
    distance_mask=0,
    distance_type="l1",
):
    """
    Performs non-parametric marker-based segmentation - useful if many detected 
    contours are touching or overlapping with each other. Input image should be 
    a binary image that serves as the true background. Iteratively, edges are 
    eroded, the difference serves as markers.

    Parameters
    ----------
    obj_input : array or container
        input object
    image_thresh: array, optional
        thresholded image n
    kernel_size: int, optional
        size of the diff-kernel (has to be odd - even numbers will be ceiled)
    iterations : int, optional
        number of times to apply diff-operation
    distance_cutoff : {between 0 and 1} float, optional
        watershed distance transform cutoff (larger values discard more pixels)
    distance_mask : {0, 3, 5} int, optional
        size of distance mask - not all sizes work with all distance types (will
        be coerced to 0)
    distance_type : {"user", "l1", "l2", "C", "l12", "fair", "welsch", "huber"} str, optional
        distance transformation type

    Returns
    -------
    image : array or container
        binary image

    """

    ##kwargs
    distance_type_list = {
        "user": cv2.DIST_USER,
        "l1": cv2.DIST_L1,
        "l2": cv2.DIST_L2,
        "C": cv2.DIST_C,
        "l12": cv2.DIST_L12,
        "fair": cv2.DIST_FAIR,
        "welsch": cv2.DIST_WELSCH,
        "huber": cv2.DIST_HUBER,
    }
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
        thresh = copy.deepcopy(image_thresh)
    elif obj_input.__class__.__name__ == "container":
        thresh = copy.deepcopy(obj_input.image)
        image = copy.deepcopy(obj_input.image_copy)

    if thresh.__class__.__name__ == "NoneType":
        print("No thresholded version of image provided for watershed - aborting.")
        return
    if len(thresh.shape) == 3:
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    ## sure background
    ## note: sure_bg is set as the thresholded input image
    sure_bg= morphology(
        thresh,
        operation="dilate",
        shape="ellipse",
        kernel_size=kernel_size,
        iterations=iterations,
    )
    
    ## sure foreground
    if distance_type in ["user", "l12", "fair", "welsch", "huber"]:
        distance_mask = 0
    opened = morphology(
        thresh,
        operation="erode",
        shape="ellipse",
        kernel_size=kernel_size,
        iterations=iterations,
    )
    dist_transform = cv2.distanceTransform(
        opened, distance_type_list[distance_type], distance_mask
    )

    dist_transform = cv2.normalize(
        dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX
    )
    
    dist_transform = blur(dist_transform, kernel_size=int(2*kernel_size))
    dist_transform = abs(255 * (1-dist_transform)) 
    dist_transform = dist_transform.astype(np.uint8)

    sure_fg = threshold(dist_transform, method="binary", value=int(distance_cutoff*255))

    ## finding unknown region
    sure_fg = sure_fg.astype("uint8")
    sure_fg[sure_fg == 1] = 255
    unknown = cv2.subtract(sure_bg, sure_fg)

    ## marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ## watershed
    markers = cv2.watershed(blur(image, int(2*kernel_size)), markers)

    ## convert to contours
    watershed_mask = np.zeros(image.shape[:2], np.uint8)
    watershed_mask[markers == -1] = 255
    watershed_mask[0 : watershed_mask.shape[0], 0] = 0
    watershed_mask[0 : watershed_mask.shape[0], watershed_mask.shape[1] - 1] = 0
    watershed_mask[0, 0 : watershed_mask.shape[1]] = 0
    watershed_mask[watershed_mask.shape[0] - 1, 0 : watershed_mask.shape[1]] = 0

    contours = find_contours(watershed_mask, retrieval="ccomp")
    image_watershed = np.zeros(watershed_mask.shape, np.uint8)
    
    for index, row in contours.iterrows():
        if row["order"] == "child":
            cv2.drawContours(
                image=image_watershed,
                contours=[row["coords"]],
                contourIdx=0,
                thickness=-1,
                color=colours["white"],
                maxLevel=3,
                offset=None
                )
            cv2.drawContours(
                image=image_watershed,
                contours=[row["coords"]],
                contourIdx=0,
                thickness=2,
                color=colours["black"],
                maxLevel=3,
                offset=None
                )
    
    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image_watershed
    elif obj_input.__class__.__name__ == "container":
        obj_input.image = image_watershed
