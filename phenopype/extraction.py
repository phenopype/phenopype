#%% modules
import cv2
import copy
import numpy as np
import numpy.ma as ma

from phenopype.settings import colours
from phenopype.utils_lowlevel import _auto_line_thickness, _load_image

#%% methods

# obj_input = p1.PC

def colour_values(obj_input, **kwargs):
    
    ## kwargs
    channels = kwargs.get("channels", ["gray"])
    
    ## load image and contours
    image, flag_input = _load_image(obj_input, load="raw")
    if flag_input == "pype_container":
        contour_binder = obj_input.contour_binder
        contour_df = obj_input.contour_df

    ## create forgeround mask
    image_bin = np.zeros(image.shape[:2], np.uint8)
    for label, contour in contour_binder.items():
        image_bin = cv2.fillPoly(image_bin, [contour["contour_points"]], colours.white)
    foreground_mask = np.array(image_bin, dtype=np.bool)

    ## method
    if "gray" in channels:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contour_df["gray_sd"], contour_df["gray_mean"] = "NA", "NA"
        for label, contour in contour_binder.items():
            rx,ry,rw,rh = cv2.boundingRect(contour["contour_points"])
            grayscale =  ma.array(data=image_gray[ry:ry+rh,rx:rx+rw], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            contour_df.loc[label]["gray_sd"] = np.std(grayscale)
            contour_df.loc[label]["gray_mean"]  = np.mean(grayscale)
            
    if "rgb" in channels:
        contour_df["red_sd"], contour_df["green_sd"], contour_df["blue_sd"] = "NA", "NA","NA"
        contour_df["red_mean"], contour_df["green_mean"], contour_df["blue_mean"] = "NA", "NA","NA"
        for label, contour in contour_binder.items():
            rx,ry,rw,rh = cv2.boundingRect(contour["contour_points"])
            blue =  ma.array(data=image[ry:ry+rh,rx:rx+rw,0], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            green =  ma.array(data=image[ry:ry+rh,rx:rx+rw,1], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            red =  ma.array(data=image[ry:ry+rh,rx:rx+rw,2], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            contour_df.loc[label]["blue_sd"]  = np.std(blue)
            contour_df.loc[label]["blue_mean"] = np.mean(blue)
            contour_df.loc[label]["green_sd"]  = np.std(green)
            contour_df.loc[label]["green_mean"] = np.mean(green)
            contour_df.loc[label]["red_sd"]  = np.std(red)
            contour_df.loc[label]["red_mean"] = np.mean(red)
            

    
    
            #                 # DRAW TO ROI
            #                 q=kwargs.get("roi_size",300)/2
            #                 if label==True:
            #                     cv2.putText(self.image_processed,  str(idx) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),5,cv2.LINE_AA)
            #                 cv2.drawContours(self.image_processed, [cnt], 0, colours.red, int(10 * self.resize_factor))
            #                 if any("skeletonize" in o for o in self.operations):                    
            #                     cv2.drawContours(self.image_processed, [skel_contour], 0, colours.green, 2)
    
            #         else:
            #             idx_noise += 1
            # else: 
            #     print("No objects found - change parameters?")