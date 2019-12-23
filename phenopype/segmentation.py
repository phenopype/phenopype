#%% modules
import cv2
import copy
import math
import numpy as np

from phenopype.settings import colours
from phenopype.utils_lowlevel import _auto_line_thickness

from phenopype.preprocessing import show_mask

#%% settings

inf = math.inf

#%% methods

def blur(obj_input, **kwargs):
    ## load image
    if isinstance(obj_input, str):
        image = cv2.imread(obj_input)  
    elif obj_input.__class__.__name__ == "pype_container":
        on = kwargs.get("on", "gray")
        if on == "bin":
            image = obj_input.image_bin
        elif on == "raw":
            image = obj_input.image_mod
        elif on == "gray":
            image = obj_input.image_gray
    else:
        image = obj_input
        
    ## kwargs
    method = kwargs.get("method","averaging")
    kernel_size = kwargs.get("kernel_size",5)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    sigma_color = kwargs.get("sigma_color",75)
    sigma_space = kwargs.get("sigma_space",75)

    ## method
    if method=="averaging":
        image_mod = cv2.blur(image,(kernel_size,kernel_size))
    elif method=="gaussian":
        image_mod = cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
    elif method=="median":
        image_mod = cv2.medianBlur(image,kernel_size)
    elif method=="bilateral":
        image_mod = cv2.bilateralFilter(image,kernel_size,sigma_color,sigma_space)
    else:
        image_mod = image
        
    ## return
    if obj_input.__class__.__name__ == "pype_container":
        if on == "bin":
            obj_input.image_bin = image_mod
        elif on == "raw":
            obj_input.image_mod = image_mod
        elif on == "gray":
            obj_input.image_gray = image_mod
        return obj_input
    else:
        return image_mod
        

def find_contours(obj_input, **kwargs):
    ## load image
    if obj_input.__class__.__name__ == "pype_container":
        image = obj_input.image_bin
    
    ## kwargs
    retr = kwargs.get("retrieval", "ext")
    retr_alg = {"ext": cv2.RETR_EXTERNAL, 
                "list": cv2.RETR_LIST, 
                "tree": cv2.RETR_TREE, 
                "ccomp": cv2.RETR_CCOMP,
                "flood": cv2.RETR_FLOODFILL}
    approx = kwargs.get("approximation", "simple")
    approx_alg = {"none": cv2.CHAIN_APPROX_NONE, 
                "simple": cv2.CHAIN_APPROX_SIMPLE, 
                "L1": cv2.CHAIN_APPROX_TC89_L1, 
                "KCOS": cv2.CHAIN_APPROX_TC89_KCOS}    
    offset_coords = kwargs.get("offset_coords", None)
    
    ## method
    image_mod, contours, hierarchy = cv2.findContours(image=image, 
                                                mode=retr_alg[retr],
                                                method=approx_alg[approx],
                                                offset=offset_coords)  
    ## return
    if obj_input.__class__.__name__ == "pype_container":
        obj_input.image_mod = image_mod
        obj_input.contour_list.append(contours)
        obj_input.contour_hierarchy.append(hierarchy)
        return obj_input
    else:
        return image_mod, contours, hierarchy


def draw_contours(obj_input, **kwargs):
    ## load image
    if isinstance(obj_input, str):
        image = cv2.imread(obj_input)       
    elif obj_input.__class__.__name__ == "pype_container":
        image = copy.deepcopy(obj_input.image)
        contour_list = obj_input.contour_list
    else:
        image = obj_input
     
    ## kwargs
    try:
        contour_list
    except NameError:
        contour_list = kwargs.get("contour_list")
    offset_coords = kwargs.get("offset_coords", None)
    thickness = kwargs.get("thickness", _auto_line_thickness(image))
    level = kwargs.get("level", 3)
    colour = eval("colours." + kwargs.get("colour", "red"))
    idx = kwargs.get("idx", 0)

    ## method
    if any(isinstance(i, list) for i in contour_list):
        contours = []
        for sublist in contour_list:
            for item in sublist:
                contours.append(item)
    else:
        contours = contour_list
    for cnt in contours:
        image = cv2.drawContours(image=image, 
                        contours=[cnt], 
                        contourIdx = idx,
                        thickness=thickness, 
                        color=colour, 
                        maxLevel=level,
                        offset=offset_coords)
        
    ## return
    image_mod = image
    if obj_input.__class__.__name__ == "pype_container":
        obj_input.image_mod = image_mod
        return obj_input
    else:
        return image_mod
    
    
def filter_contours(obj_input, **kwargs):
    ## load image
    if isinstance(obj_input, str):
        image = cv2.imread(obj_input)  
    elif obj_input.__class__.__name__ == "pype_container":
        contour_list_all = obj_input.contour_list

    ## kwargs
    min_nodes, max_nodes = kwargs.get('min_nodes', 5), kwargs.get('max_nodes', inf)
    min_diameter, max_diameter = kwargs.get('min_diameter', 0), kwargs.get('max_diameter', inf)
    min_area, max_area = kwargs.get('min_area', 0), kwargs.get('max_area', inf)

    ## method
    if contour_list_all:
        contour_list_all_new = []
        for contour_list in contour_list_all:
            contour_list_new = []
            for cnt in contour_list:

                ## quality check
                if all([
                        ## number of contour nodes
                        len(cnt) > min_nodes and len(cnt) < max_nodes,
                        ## contour diameter
                        int(cv2.minEnclosingCircle(cnt)[1]*2) > min_diameter and int(cv2.minEnclosingCircle(cnt)[1]*2) < max_diameter,
                        ## contour area
                        int(cv2.contourArea(cnt)) > min_area and int(cv2.contourArea(cnt)) < max_area
                        ]):
                    contour_list_new.append(cnt)
            contour_list_all_new.append(contour_list_new)
                
    else:
        contour_list_all_new = contour_list_all

    if obj_input.__class__.__name__ == "pype_container":
        obj_input.contour_list = contour_list_all_new
        # obj_input.contour_hierarchy =
        return obj_input
    else:
        return contour_list_all_new
    
    
    
def morphology(obj_input, **kwargs):
    ## load image
    if isinstance(obj_input, str):
        image = cv2.imread(obj_input)
    elif obj_input.__class__.__name__ == "pype_container":
        on = kwargs.get("on", "bin")
        if on == "bin":
            image = obj_input.image_bin
        elif on == "raw":
            image = obj_input.image_mod
        elif on == "gray":
            image = obj_input.image_gray
    else:
        image = obj_input
    
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
    
    ## method
    image_mod = cv2.morphologyEx(image, 
                                 op=operation_list[operation], 
                                 kernel = kernel,
                                 iterations = iterations)

    ## return
    if obj_input.__class__.__name__ == "pype_container":
        if on == "bin":
            obj_input.image_bin = image_mod
        elif on == "raw":
            obj_input.image_mod = image_mod
        elif on == "gray":
            obj_input.image_gray = image_mod
        return obj_input
    else:
        return image_mod



def threshold(obj_input, **kwargs):
    ## load image
    if isinstance(obj_input, str):
        image = cv2.imread(obj_input)       
    elif obj_input.__class__.__name__ == "pype_container":
        image = obj_input.image_gray
    else:
        image = obj_input
    
    ## kwargs
    method = kwargs.get("method", "otsu")
    blocksize = kwargs.get("blocksize", 99)
    constant = kwargs.get("constant", 1)
    value = kwargs.get("value", 127)
    
    ## method
    if method == "otsu":
        ret, image_mod = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        image_mod = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blocksize, constant)
    elif method == "binary":
        ret, image_mod = cv2.threshold(image, value, 255,cv2.THRESH_BINARY_INV)  
    else:
        image_mod = image
        
    ## apply mask
    if len(obj_input.mask_binder)>0:
        for key, value in obj_input.mask_binder.items():
            MO = value
            if MO.include == True:
                image_mod[np.invert(MO.mask_bool)] = 0
    
    ## return
    if obj_input.__class__.__name__ == "pype_container":
        obj_input.image_bin = image_mod
        return obj_input
    else:
        return image_mod
    
    
    
    