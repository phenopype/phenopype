#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from math import inf

from phenopype.settings import *
from phenopype.core.preprocessing import invert_image
from phenopype.utils_lowlevel import _create_mask_bin, _create_mask_bool, _image_viewer, _auto_line_width

#%% functions

def blur(obj_input, **kwargs):
    """
    

    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    ## kwargs
    method = kwargs.get("method","averaging")
    kernel_size = kwargs.get("kernel_size",5)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    sigma_color = kwargs.get("sigma_color",75)
    sigma_space = kwargs.get("sigma_space",75)

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image

    ## method
    if method=="averaging":
        image = cv2.blur(image,(kernel_size,kernel_size))
    elif method=="gaussian":
        image = cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
    elif method=="median":
        image = cv2.medianBlur(image,kernel_size)
    elif method=="bilateral":
        image = cv2.bilateralFilter(image,kernel_size,sigma_color,sigma_space)
    else:
        image = image

    ## return
    if obj_input.__class__.__name__ == "container":
        obj_input.image = image
    else:
        return image



def draw(obj_input, **kwargs):
    """
    

    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ## kwargs
    flag_overwrite = kwargs.get("overwrite", False)
    flag_tool = kwargs.get("tool", "line")
    line_col = kwargs.get("colour", "black")

    ## load image
    df_draw, df_image_data = None, None
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"})
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_draw"):
            df_draw = obj_input.df_draw
    else:
        print("wrong input format.")
        return

    ## more kwargs
    if flag_tool in ["rect", "rectangle", "poly", "polygon"]:
        line_width = -1
    else:
        line_width = kwargs.get("line_width", max(1,_auto_line_width(image)))

    while True:
        ## check if exists
        if not df_draw.__class__.__name__ == "NoneType" and flag_overwrite == False:
            print("- polylines already drawn (overwrite=False)")
            break
        elif not df_draw.__class__.__name__ == "NoneType" and flag_overwrite == True:
            print("- draw polylines (overwriting)")
            pass
        elif not df_draw.__class__.__name__ == "NoneType" and flag_edit == True:
            print("- draw polylines (editing)")
            pass
        elif df_draw.__class__.__name__ == "NoneType":
            print("- draw polylines")
            pass
        
        ## method
        out = _image_viewer(image, 
                            tool=flag_tool, 
                            draw=True,
                            line_width=line_width,
                            line_col=line_col)
        
        ## abort
        if not out.done:
            if obj_input.__class__.__name__ == "ndarray":
                print("terminated polyline creation")
                return 
            elif obj_input.__class__.__name__ == "container":
                print("- terminated polyline creation")
                return True

        ## create df
        df_draw = pd.DataFrame({"tool": flag_tool}, index=[0])
        df_draw["line_width"] = line_width
        df_draw["colour"] = line_col
        df_draw["coords"] = str(out.point_list)
        
        break

    ## draw
    for idx, row in df_draw.iterrows():
        coord_list = eval(row["coords"])
        for coords in coord_list:
            if row["tool"] in ["line", "lines"]:
                cv2.polylines(image, np.array([coords]), False, colours[row["colour"]], row["line_width"])
            elif row["tool"] in ["rect", "rectangle"]:
                cv2.fillPoly(image, np.array([coords]), colours[row["colour"]])

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        df_draw = pd.concat([pd.concat([df_image_data]*len(df_draw)).reset_index(drop=True), 
                        df_draw.reset_index(drop=True)], axis=1)
        return image, df_draw
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_draw = df_draw
        obj_input.image = image



def find_contours(obj_input, **kwargs):
    """
    

    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    ## kwargs
    df_image_data = kwargs.get("df", None)
    flag_ret_cont = kwargs.get("return_contours",True)
    retr = kwargs.get("retrieval", "ext")
    retr_alg = {"ext": cv2.RETR_EXTERNAL, ## only external
                "list": cv2.RETR_LIST, ## all contours
                "tree": cv2.RETR_TREE, ## fully hierarchy
                "ccomp": cv2.RETR_CCOMP, ## outer perimeter and holes
                "flood": cv2.RETR_FLOODFILL} ## not sure what this does
    approx = kwargs.get("approximation", "simple")
    approx_alg = {"none": cv2.CHAIN_APPROX_NONE, ## no approximation of the contours, all points
                "simple": cv2.CHAIN_APPROX_SIMPLE,  ## minimal corners
                "L1": cv2.CHAIN_APPROX_TC89_L1, ## algorithm 1
                "KCOS": cv2.CHAIN_APPROX_TC89_KCOS} ## algorithm 2
    offset_coords = kwargs.get("offset_coords", (0,0))
    min_nodes, max_nodes = kwargs.get('min_nodes', 3), kwargs.get('max_nodes', inf)
    min_diameter, max_diameter = kwargs.get('min_diameter', 0), kwargs.get('max_diameter', inf)
    min_area, max_area = kwargs.get('min_area', 0), kwargs.get('max_area', inf)

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if df_image_data.__class__.__name__ == "NoneType":
            df_image_data = pd.DataFrame({"filename":"unknown"}, index=[0])
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image
        df_image_data = obj_input.df_image_data
    else:
        print("wrong input format.")
        return

    ## check
    if len(image.shape)>2:
        print("Multi-channel array supplied - need binary array.")

    ## method
    image, contour_list, hierarchy = cv2.findContours(image=image, 
                                                mode=retr_alg[retr],
                                                method=approx_alg[approx],
                                                offset=offset_coords)

    ## filtering
    if not contour_list.__class__.__name__ == "NoneType" and len(contour_list)>0:
        contour_dict = {}
        idx = 0
        for contour, hier in zip(contour_list, hierarchy[0]):
            
            ## number of contour nodes
            if len(contour) > min_nodes and len(contour) < max_nodes:
                center, radius = cv2.minEnclosingCircle(contour)
                center = int(center[0]), int(center[1])
                diameter = int(radius*2)
                area = int(cv2.contourArea(contour))
                if hier[3] == -1:
                    cont_order = "parent"
                else:
                    cont_order = "child"
                if all([
                    diameter > min_diameter and diameter < max_diameter,
                    area > min_area and area < max_area,
                    ]):
                        idx += 1
                        contour_label = str(idx)
                        contour_dict[contour_label] = {"contour":contour_label, 
                                                       "center": center,
                                                       "diameter": diameter, 
                                                       "area":area,
                                                       "order": cont_order,
                                                       "idx_child":hier[2],
                                                       "idx_parent":hier[3],
                                                       "coords":contour}
    else:
        print("No contours found.")
        
    ## output
    df_contours = pd.DataFrame(contour_dict).T
    df_contours.reset_index(drop=True, inplace=True)
    df_contours = pd.concat([pd.concat([df_image_data]*len(df_contours)).reset_index(drop=True), 
                             df_contours.reset_index(drop=True)], axis=1)

    if obj_input.__class__.__name__ == "ndarray":
        return  df_contours
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_contours = df_contours



def morphology(obj_input, kernel_size=5, shape="rect", operation="close", 
               iterations=1):
    """
    

    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    kernel_size : TYPE, optional
        DESCRIPTION. The default is 5.
    shape : TYPE, optional
        DESCRIPTION. The default is "rect".
    operation : TYPE, optional
        DESCRIPTION. The default is "close".
    iterations : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    ## kwargs
    shape_list = {"cross": cv2.MORPH_CROSS, 
                  "rect": cv2.MORPH_RECT, 
                  "ellipse": cv2.MORPH_ELLIPSE}
    operation_list = {"erode": cv2.MORPH_ERODE, 
                      "dilate": cv2.MORPH_DILATE,
                      "open": cv2.MORPH_OPEN, 
                      "close": cv2.MORPH_CLOSE, 
                      "gradient": cv2.MORPH_GRADIENT,
                      "tophat ": cv2.MORPH_TOPHAT, 
                      "blackhat": cv2.MORPH_BLACKHAT, 
                      "hitmiss": cv2.MORPH_HITMISS}  
    kernel = cv2.getStructuringElement(shape_list[shape], (kernel_size, kernel_size))

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image
    
    ## method
    image = cv2.morphologyEx(image, 
                                 op=operation_list[operation], 
                                 kernel = kernel,
                                 iterations = iterations)

    ## return
    if obj_input.__class__.__name__ == "container":
        obj_input.image = image
    else:
        return image



def skeletonize(img):
    """
    

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    skeleton : TYPE
        DESCRIPTION.
    iters : TYPE
        DESCRIPTION.

    """

    skeleton = np.zeros(img.shape,np.uint8)
    eroded = np.zeros(img.shape,np.uint8)
    temp = np.zeros(img.shape,np.uint8)

    _,thresh = cv2.threshold(img,127,255,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton,iters)



def threshold(obj_input, df_masks=None, method="otsu", constant=1, blocksize=99, 
              value=127, channel="gray", invert=False):
    """
    If the input array was single channel, the threshold method can only use the 
    grayscale space to, but if multiple channels were provided, then one can either chose 
    to coerce the color image to grayscale or use one of the color channels directly.  



    """


    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = copy.deepcopy(obj_input)
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image
        if hasattr(obj_input, "df_masks"):
            df_masks = copy.deepcopy(obj_input.df_masks)

    ## channel
    if len(image.shape)==3:
        if channel == "gray" or channel=="grayscale":
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        elif channel == "g" or channel== "green":
            image = image[:,:,0]
        elif channel == "r" or channel== "red":
            image = image[:,:,1]
        elif channel == "blue" or channel== "b":
            image = image[:,:,2]

    if invert:
        image = invert_image(image)

    ## method
    if method == "otsu":
        ret, image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        image = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blocksize, constant)
    elif method == "binary":
        ret, image = cv2.threshold(image, value, 255,cv2.THRESH_BINARY_INV)

    ## apply masks
    if not df_masks.__class__.__name__ == "NoneType":
        mask_bool = np.zeros(image.shape, dtype=bool)
        for index, row in df_masks.iterrows():
            coords = eval(row["coords"])
            if not row["mask"] == "":
                label = row["mask"]
                print("- applying mask: " + label)
            if row["include"]:
                mask_bool = np.logical_or(mask_bool, _create_mask_bool(image, coords))
            if not row["include"]:
                image[_create_mask_bool(image, coords)] = 0
        image[mask_bool==0] = 0

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.image = image
        obj_input.image_bin = image



def watershed(obj_input, close_iterations=3, close_kernel_size=3, 
              open_iterations=3, open_kernel_size=3, distance_cutoff=0.5,
              distance_mask=0, distance_type="l1", shape="ellipse", **kwargs):
    """
    If the input array was single channel, the threshold method can only use the 
    grayscale space to, but if multiple channels were provided, then one can either chose 
    to coerce the color image to grayscale or use one of the color channels directly.  


    """

    ##kwargs
    distance_type_list = {"user": cv2.DIST_USER , 
                          "l1": cv2.DIST_L1,
                          "l2": cv2.DIST_L2, 
                          "C": cv2.DIST_C, 
                          "l12": cv2.DIST_L12,
                          "fair": cv2.DIST_FAIR, 
                          "welsch": cv2.DIST_WELSCH, 
                          "huber": cv2.DIST_HUBER}  

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        thresh = copy.deepcopy(obj_input.image)
        image = copy.deepcopy(obj_input.image_copy)

    if len(thresh.shape)==3:
        thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)

    ## sure background 
    ## note: sure_bg is set as the thresholded input image
    sure_bg =  copy.deepcopy(thresh)
    
    ## sure foreground 
    if distance_type in ["user","l12", "fair", "welsch", "huber"]:
        distance_mask = 0
    opened = morphology(thresh, operation="open", shape=shape, kernel_size=open_kernel_size, iterations=open_iterations)
    dist_transform = cv2.distanceTransform(opened,distance_type_list[distance_type],distance_mask)
    dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    ret, sure_fg = cv2.threshold(dist_transform,distance_cutoff,1,0)

    ## finding unknown region
    sure_fg = sure_fg.astype("uint8")
    sure_fg[sure_fg==1] = 255
    unknown = cv2.subtract(sure_bg,sure_fg)

    ## marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    
    ## watershed
    markers = cv2.watershed(image, markers)
    image = np.zeros(image.shape[:2], np.uint8)
    image[markers == -1] = 255
    
    ## convert to contours
    markers1 = markers.astype(np.uint8)
    ret, image = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    image[0:image.shape[0], 0] = 0
    image[0:image.shape[0], image.shape[1]-1] = 0
    image[0, 0:image.shape[1]] = 0
    image[image.shape[0]-1,  0:image.shape[1]] = 0

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.image = image
