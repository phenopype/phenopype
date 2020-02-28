#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from math import inf

from phenopype.utils_lowlevel import _create_mask_bin, _create_mask_bool #, _load_masks

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
            df_image_data = pd.DataFrame({"filename":"unknown"})
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.canvas
        df_image_data = obj_input.df_image_data
        if hasattr(obj_input, "df_contours"):
            df_contours = obj_input.df_contours
    else:
        warnings.warn("wrong input format.")
        return

    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image

    ## check
    if len(image.shape)>2:
        warnings.warn("Multi-channel array supplied - need binary array.")

    ## method
    image, contour_list, hierarchy = cv2.findContours(image=image, 
                                                mode=retr_alg[retr],
                                                method=approx_alg[approx],
                                                offset=offset_coords)

    ## filtering
    if contour_list:
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
        warnings.warn("No contours found.")
        
    ## output
    df_contours = pd.DataFrame(contour_dict).T
    df_contours.reset_index(drop=True, inplace=True)
    df_contours = pd.concat([pd.concat([df_image_data]*len(df_contours)).reset_index(drop=True), 
                             df_contours.reset_index(drop=True)], axis=1)


    if obj_input.__class__.__name__ == "ndarray":
        return  df_contours
    elif obj_input.__class__.__name__ == "container":
        obj_input.df_contours = df_contours

def morphology(obj_input, **kwargs):
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
    kernel_size = kwargs.get("kernel_size", 5)
    shape = kwargs.get("shape", "rect")
    shape_list = {"cross": cv2.MORPH_CROSS, 
                "rect": cv2.MORPH_RECT, 
                "ellipse": cv2.MORPH_ELLIPSE}
    kernel = cv2.getStructuringElement(shape_list[shape], (kernel_size, kernel_size))
    operation = kwargs.get("operation", "close")
    operation_list = {"erode": cv2.MORPH_ERODE, 
                      "dilate": cv2.MORPH_DILATE,
                      "open": cv2.MORPH_OPEN, 
                      "close": cv2.MORPH_CLOSE, 
                      "gradient": cv2.MORPH_GRADIENT,
                      "tophad ": cv2.MORPH_TOPHAT, 
                      "blackhat": cv2.MORPH_BLACKHAT, 
                      "hitmiss": cv2.MORPH_HITMISS}  
    iterations = kwargs.get("iterations", 1)
    
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



def threshold(obj_input, **kwargs):
    """
    If the input array was single channel, the threshold method can only use the 
    grayscale space to, but if multiple channels were provided, then one can either chose 
    to coerce the color image to grayscale or use one of the color channels directly.  


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
    blocksize = kwargs.get("blocksize", 99)
    constant = kwargs.get("constant", 1)
    colourspace = kwargs.get("colourspace", "gray")
    method = kwargs.get("method", "otsu")
    value = kwargs.get("value", 127)
    df_masks = kwargs.get("masks", None)
    
    ## load image
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image
        if hasattr(obj_input, "df_masks"):
            df_masks = copy.deepcopy(obj_input.df_masks)

    ## colourspace
    if len(image.shape)==3:
        if colourspace == "gray" or colourspace=="grayscale":
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        elif colourspace == "g" or colourspace== "green":
            image = image[:,:,0]
        elif colourspace == "r" or colourspace== "red":
            image = image[:,:,1]
        elif colourspace == "blue" or colourspace== "b":
            image = image[:,:,2]
            
    ## method
    if method == "otsu":
        ret, image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        image = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blocksize, constant)
    elif method == "binary":
        ret, image = cv2.threshold(image, value, 255,cv2.THRESH_BINARY_INV)  

    ## apply masks
    if not df_masks.__class__.__name__ == "NoneType":
        mask_bool = np.zeros(image.shape, np.bool)
        label = ""
        for index, row in df_masks.iterrows():
            coords = eval(row["coords"])
            if not row["mask"] == label:
                label = row["mask"]
                print(" - applying mask: " + label)
            if row["include"]:
                mask_bool = np.logical_or(mask_bool, _create_mask_bool(image, coords))
            if not row["include"]:
                mask_bool[_create_mask_bool(image, coords)] = False
        image[mask_bool==0] = 0

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return image
    elif obj_input.__class__.__name__ == "container":
        obj_input.image = image
        obj_input.image_bin = image
