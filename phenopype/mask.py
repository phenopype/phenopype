import os
import cv2
import numpy as np
import exifread
from collections import Counter

from phenopype.utils_lowlevel import _image_viewer

#%%

class create:
    """Intialize mask maker, loads image.
    
    Parameters
    ----------

    image: str or array
        absolute or relative path to OR numpy array of image 
    """        
    def __init__(self, image, **kwargs):
        # initialize # ----------------
        
        if isinstance(image, str):
            image = cv2.imread(image)
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        self.image = image
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.idx = 0

        ## kwargs 
        self.zoom_fac = kwargs.get("zoom_factor", 5)                       
        self.mode = kwargs.get("mode", "rectangle")                       
        self.show = kwargs.get("show", "True")          

        ## initialize with empty lists / parameters
        self.done = False 
        self.flag_zoom = False
        self.current = (0, 0) 
        self.current_zoom = []
        self.points = []
        self.idx = 0
        self.idx_list = []
        
        self.overlay = np.zeros(self.image.shape, np.uint8) # make overlay
        self.overlay[:,:,2] = 200 # start with all-red overlay
        
        ## for zoom
        self.image_height = self.template_image.shape[0]
        self.delta_height = int((self.image_height/self.zoom_fac)/2)
        self.image_width = self.template_image.shape[1]
        self.delta_width = int((self.image_width/self.zoom_fac)/2)
        self.image_diag = int((self.image_height + self.image_width)/2)
        
#    def _on_mouse(self, event, x, y, buttons, user_param):
#        if self.done: # Nothing more to do
#            return
#        if event == cv2.EVENT_MOUSEMOVE:
#            self.current = (x, y)
#        if event == cv2.EVENT_LBUTTONDOWN:
#            self.points.append((x, y))
#            self.idx += 1
#            print("Adding point #%d with position(%d,%d) to arena" % (self.idx, x, y))
#
#        if event == cv2.EVENT_RBUTTONDOWN:
#            if len(self.points) > 0:
#                self.points = self.points[:-1]
#                self.idx -= 1
#                print("Removing point #%d with position(%d,%d) from arena" % (self.idx, x, y))
#            else:
#                print("No points to delete")
                
    def _on_mouse(self, event, x, y, flags, params):      
        """- Internal reference - don't call this directly -  
        
        Mouse events for "make_scale_template" function.
        """ 
        if self.done: 
            return       
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)                
        if event == cv2.EVENT_MOUSEWHEEL:               
            if flags > 0 and self.flag_zoom == False:
                self.flag_zoom=True
                self.current_zoom = (x,y)
            if flags < 0 and self.flag_zoom == True:
                self.flag_zoom=False
        if self.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.flag_zoom == True:
                    x_zoom, y_zoom = self.current_zoom     
                    x = x + max(0,x_zoom-self.delta_width)
                    y = y + max(0,y_zoom-self.delta_height)
                self.points.append((x, y))
                self.idx += 1  
                self.idx_list.append(self.idx)
                print("Adding point #%d with position(%d,%d) to polygon" % (self.idx, x, y))
            if event == cv2.EVENT_RBUTTONDOWN:
                if len(self.points) > 0:
                    self.points = self.points[:-1]
                    self.idx -= 1
                    self.idx_list = self.idx_list[:-1]
                    print("Removing point #%d with position(%d,%d) from polygon" % (self.idx, x, y))
                else:
                    print("No point to delete")
        if self.mode == "rectangle":
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.flag_zoom == True:
                    x_zoom, y_zoom = self.current_zoom     
                    x = x + max(0,x_zoom-self.delta_width)
                    y = y + max(0,y_zoom-self.delta_height)
                if len(self.points) < 2:
                    self.points.append((x, y))
                    self.idx += 1  
                    self.idx_list.append(self.idx)
                    print("Adding point %d/2 with position(%d,%d) to rectangle" % (self.idx, x, y))
                elif len(self.points) == 2:
                    print("Finished - no more points to add!")
            if event == cv2.EVENT_RBUTTONDOWN:
                if len(self.points) > 0:
                    self.points = self.points[:-1]
                    self.idx -= 1
                    self.idx_list = self.idx_list[:-1]
                    print("Removing point %d with position(%d,%d) from rectangle" % (self.idx, x, y))
                else:
                    print("No point to delete")
                
    def create(self, **kwargs):
        """Mask maker method to draw rectangle or polygon mask onto image.
        
        Parameters
        ----------        
        
        include: bool (default: True)
            determine whether resulting mask is to include or exclude objects within
        label: str (default: "area1")
            passes a label to the mask
        mode: str (default: "rectangle")
            zoom into the scale with "rectangle" or "polygon".
            
        """
        include = kwargs.get("include",True)
        mode = kwargs.get("mode","rectangle")
        label = kwargs.get("label","area 1")
        
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self._on_mouse)
        
        

        
        if not "show" in vars(self):
            temp_canvas1 = copy.deepcopy(self.image)
            temp_canvas2 = copy.deepcopy(self.image)

        else:
            temp_canvas1 = copy.deepcopy(self.show)
            temp_canvas2 = copy.deepcopy(self.show)

        print("\nMark the outline of your arena, i.e. what you want to include in the image analysis by left clicking, finish with enter.")
        
        if mode == "new":
        
            while(not self.done):
            
                ## read key input
                k = cv2.waitKey(50)
    
                ## mousewheel zoom
                self.points_new = copy.deepcopy(self.points)
                if self.flag_zoom == True:   
                    
                    ## zoom rectangle
                    x_zoom, y_zoom = self.current_zoom
                    x_res1 = max(0,x_zoom-self.delta_width)
                    x_res2 = x_zoom+self.delta_width
                    y_res1 = max(0,y_zoom-self.delta_height)
                    y_res2 = y_zoom+self.delta_height
                    temp_canvas2 = temp_canvas2[y_res1:y_res2,x_res1:x_res2]
                    
                    ## update points to draw in zoom rectangle                
                    idx = -1
                    for i in self.points_new:
                        idx += 1
                        x, y = i
                        x = x - x_res1
                        y = y - y_res1
                        self.points_new[idx] = (x,y)    

                            
        # =============================================================================
        # draw rectangle 
        # =============================================================================             
                
        if mode == "rectangle":
            (x,y,w,h) = cv2.selectROI("phenopype", temp_canvas1, fromCenter=False)
            if cv2.waitKey(50) & 0xff == 13:
                cv2.destroyWindow("phenopype")
                self.done = True
            elif cv2.waitKey(50) & 0xff == 27:
                cv2.destroyWindow("phenopype")  
                self.done = True
            self.points = [(x, y), (x, y+h), (x+w, y+h), (x+w, y)]
            self.done = True
            
        # =============================================================================
        # draw polygon 
        # =============================================================================
        
        elif mode == "polygon":
            while(not self.done):
                if (len(self.points) > 0):
                    cv2.polylines(temp_canvas1, np.array([self.points]), False, green, 3)
                    cv2.line(temp_canvas1, self.points[-1], self.current, blue, 3)
                cv2.imshow("phenopype", temp_canvas1)
                temp_canvas1 = copy.deepcopy(temp_canvas2)
                if cv2.waitKey(50) & 0xff == 13:
                    self.done = True
                    cv2.destroyWindow("phenopype")
                elif cv2.waitKey(50) & 0xff == 27:
                    self.done = True
                    cv2.destroyWindow("phenopype")
                           
        zeros = np.zeros(self.image.shape[0:2], np.uint8)
        mask = cv2.fillPoly(zeros, np.array([self.points]), white)
        
        if include == True:
            mask_bool = np.array(mask, dtype=bool)
            self.overlay[mask_bool,1] = 200   
            self.overlay[mask_bool,2] = 0   
            line_col = green

        elif include == False:
            mask_bool = np.array(mask, dtype=bool)
            self.overlay[mask_bool,2] = 200   
            self.overlay[mask_bool,1] = 0   
            line_col = red
      
        if mode == "rectangle":
            cv2.rectangle(self.overlay,(int(self.points[0][0]),int(self.points[0][1])),(int(self.points[2][0]),int(self.points[2][1])),line_col,10)
        elif mode == "polygon":
            cv2.polylines(self.overlay, np.array([self.points]), True, line_col, 10)      
            
        cv2.putText(self.overlay ,label ,self.points[0] ,cv2.FONT_HERSHEY_SIMPLEX, 4, line_col,4 ,cv2.LINE_AA)
        self.show = cv2.addWeighted(self.image, .7, self.overlay, 0.5, 0) # combine

        self.points1 = self.points


        # reset
        self.done = False 
        self.current = (0, 0) 
        self.points = [] 
        self.idx = 0
        
        
        if kwargs.get('show', False) == True:
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("phenopype", self.show)
            cv2.waitKey(0)
            cv2.destroyWindow("phenopype")

        else:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            
        return mask_bool, label, include