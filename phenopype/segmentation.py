#%%
import cv2
import copy

from phenopype.settings import colours
from phenopype.utils import show_img
from phenopype.utils_lowlevel import _auto_line_thickness

#%%

def blur(image, **kwargs):
    ## load image
    if isinstance(image, str):
        image = cv2.imread(image)  
    
    ## kwargs
    method = kwargs.get("method","averaging")
    kernel_size = kwargs.get("kernel_size",5)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    sigma_color = kwargs.get("sigma_color",75)
    sigma_space = kwargs.get("sigma_space",75)

    ## method
    if method=="averaging":
        image_blurred = cv2.blur(image,(kernel_size,kernel_size))
    elif method=="gaussian":
        image_blurred = cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
    elif method=="median":
        image_blurred = cv2.medianBlur(image,kernel_size)
    elif method=="bilateral":
        image_blurred = cv2.bilateralFilter(image,kernel_size,sigma_color,sigma_space)
    else:
        image_blurred = image
    
    ## return
    return image_blurred

def find_contours(image, **kwargs):
    
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
    im2, contours, hierarchy = cv2.findContours(image=image, 
                                                mode=retr_alg[retr],
                                                method=approx_alg[approx],
                                                offset=offset_coords)
    return im2, contours, hierarchy


def draw_contours(image, contours, **kwargs):
    
    ## kwargs
    offset_coords = kwargs.get("offset_coords", None)
    thickness = kwargs.get("thickness", _auto_line_thickness(image))
    level = kwargs.get("level", 3)
    colour = eval("colours." + kwargs.get("colour", "red"))
    idx = kwargs.get("idx", 0)
    flag_show = kwargs.get("show", False)

    
    if isinstance(contours, list):
        pass
    else:
        contours = [contours]
    
    ## method
    image_mod = copy.deepcopy(image)
    image_mod = cv2.drawContours(image=image_mod, 
                    contours=contours, 
                    contourIdx = idx,
                    thickness=thickness, 
                    color=colour, 
                    maxLevel=level,
                    offset=offset_coords)
    
    if flag_show:    
        show_img(image_mod)
    
    return image_mod
    


def threshold(image, **kwargs):
    ## load image
    if isinstance(image, str):
        image = cv2.imread(image)  
    if len(image.shape)==3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
        
    ## kwargs
    method = kwargs.get("method", "otsu")
    blocksize = kwargs.get("blocksize", 99)
    constant = kwargs.get("constant", 1)
    value = kwargs.get("value", 127)
    
    ## method
    if method == "otsu":
        ret, image_bin = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        image_bin = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blocksize, constant)
    elif method == "binary":
        ret, image_bin = cv2.threshold(image_gray, value, 255,cv2.THRESH_BINARY_INV)  
    else:
        image_bin = image
    
    ## return
    return image_bin
    
    
    
    