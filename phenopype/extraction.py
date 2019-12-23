#%% modules
import cv2
import copy
import numpy as np
import numpy.ma as ma

from phenopype.settings import colours
from phenopype.utils_lowlevel import _auto_line_thickness

from phenopype.preprocessing import show_mask

#%% methods


def colours(obj_input, **kwargs):
    ## load image
    if isinstance(obj_input, str):
        image = cv2.imread(obj_input)  
    elif obj_input.__class__.__name__ == "pype_container":
        image = obj_input.image_mod
        image_gray = obj_input.image_gray
        image_bin = obj_input.image_bin

        contour_list_all = obj_input.contour_list
        
    ## kwargs
    channels = kwargs.get("channels", ["gray"])
    
    ## init
    
    pp.show_img(image_gray[ry:ry+rh,rx:rx+rw])

        
    ## method
    for contour_list in contour_list_all:
        for cnt in contour_list:
            rx,ry,rw,rh = cv2.boundingRect(cnt)

            if "gray" in channels:
                grayscale =  ma.array(data=image_gray[ry:ry+rh,rx:rx+rw], mask = np.logical_not(image_bin[ry:ry+rh,rx:rx+rw]))
                grayscale_mean = int(np.mean(grayscale)) 
                grayscale_sd = int(np.std(grayscale)) 

                    b =  ma.array(data=image[ry:ry+rh,rx:rx+rw,], mask = np.logical_not(thresh[ry:ry+rh,rx:rx+rw]))
                    g =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,1], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                    r =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,2], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw])))
                            bgr_mean = (int(np.mean(b)),int(np.mean(g)),int(np.mean(r))) # mean grayscale value
                            bgr_sd = (int(np.std(b)),int(np.std(g)),int(np.std(r))) # mean grayscale value
                            cnt_list = cnt_list + [bgr_mean, bgr_sd]
                            df_column_names = df_column_names + ["bgr_mean","bgr_sd"]
                                
                            df_list.append(cnt_list)    
    
    
                            # DRAW TO ROI
                            q=kwargs.get("roi_size",300)/2
                            if label==True:
                                cv2.putText(self.image_processed,  str(idx) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),5,cv2.LINE_AA)
                            cv2.drawContours(self.image_processed, [cnt], 0, colours.red, int(10 * self.resize_factor))
                            if any("skeletonize" in o for o in self.operations):                    
                                cv2.drawContours(self.image_processed, [skel_contour], 0, colours.green, 2)
    
                    else:
                        idx_noise += 1
            else: 
                print("No objects found - change parameters?")