import os
import cv2
import numpy as np
import copy
import pandas as pd 
import sys

from phenopype.utils import (show_img)


#%% colours

green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
black = (0,0,0)
white = (255,255,255)

#colours = {"red": (0, 0, 255),
# "green": (0, 255, 0), 
# "blue": (255, 0, 0),
# "black":(0,0,0),
# "white":(255,255,255)}


#%% modules
class landmark_maker:
    """Intialize landmarks_maker, loads image.
    
    Parameters
    ----------
    image: str or array
        absolute or relative path to OR numpy array of image 
    scale: num (default: 1)
        pixel to mm-ratio 
    ID: str (default: NA)
        specimen ID; "query" is special flag for user entry
    point_size: num (default: 1/300 of image diameter)
        size of the landmarks on the image in pixels
    point_col: value (default: red)
        colour of landmark (red, green, blue, black, white)
    label_size: num (1/1500 of image diamter)
        size of the numeric landmark label in pixels
    label_col: value (default: black)
        colour of label (red, green, blue, black, white)
    draw_line: bool (default: False)
        flag to draw arc and measure it's length
    zoom_factor: int (default 5)
        magnification factor on mousewheel use
    
    Returns
    -------
    
    .df = pandas data frame with landmarks (and arc-length, if selected)
    .drawn = image array with drawn landmarks (and lines)
    .ID = provided specimen ID 
    
    """        
    
    def __init__(self, image, **kwargs):
        
        # initialize # ----------------
        if isinstance(image, str):
            self.image = cv2.imread(image)
            self.filename = os.path.basename(image)
        else:
            self.image = image
            self.filename = kwargs.get("filename","NA")
                
    def _on_mouse(self, event, x, y, flags, params):
        
        if self.done: 
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)          
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.flag_zoom == True:
                x_zoom, y_zoom = self.current_zoom     
                x = x + max(0,x_zoom-self.delta_width)
                y = y + max(0,y_zoom-self.delta_height)
                self.points.append((x, y))
            else:
                self.points.append((x, y))
            self.idx += 1  
            self.idx_list.append(self.idx)            
            print("Point #%d with position (x=%d,y=%d) added" % (self.idx, x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points = self.points[:-1]
                self.idx -= 1
                self.idx_list = self.idx_list[:-1]
                print("Point #%d with position (x=%d,y=%d) deleted" % (self.idx, x, y))
            else:
                print("No point to delete")
        if event == cv2.EVENT_MOUSEWHEEL:               
            if flags > 0 and self.flag_zoom == False:
                self.flag_zoom=True
                self.current_zoom = (x,y)
            if flags < 0 and self.flag_zoom == True:
                self.flag_zoom=False

                
    def set_landmarks(self, **kwargs):                   
        
        self.ID_flag = kwargs.get("ID","NA")
        if self.ID_flag == "query":
            self.ID = ""
        else:
            self.ID = self.ID_flag
        self.scale = kwargs.get("scale", 1)
        self.zoom_fac = kwargs.get("zoom_factor", 5)
        self.draw_line = kwargs.get("draw_line", False)
        self.show = kwargs.get('show', True)
        
        self.image_height = self.image.shape[0]
        self.delta_height = int((self.image_height/self.zoom_fac)/2)
        self.image_width = self.image.shape[1]
        self.delta_width = int((self.image_width/self.zoom_fac)/2)
        self.image_diag = int((self.image_height + self.image_width)/2)
        self.image_diag_fac = int(self.image_diag/10)
        
        self.point_size = kwargs.get("point_size", int(self.image_diag/300))
        self.point_col = kwargs.get("point_col", red)
        self.label_size = kwargs.get("label_size", int(self.image_diag/1750))
        self.label_col = kwargs.get("label_col", black)

        self.done = False 
        self.current = (0, 0) 
        self.current_zoom = []
        self.points = []
        self.points_zoom = []
        self.idx = 0
        self.idx_list = []
        self.flag_zoom = False

        # =============================================================================
        # add landmarks
        # =============================================================================
        
        print("\nAdd landmarks by left clicking, remove by right clicking, finish with enter.")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self._on_mouse)
        
        temp_canvas1 = copy.deepcopy(self.image)
        temp_canvas2 = temp_canvas1
        
        self.done = False 
        self.current = (0, 0) 
        self.idx = 0
        self.idx_list = []
        self.points = []
        
        while(not self.done):
            
            ## read key input
            k = cv2.waitKey(50)
            
            # =============================================================================
            # if mousewheel-zoom, update coordinate-space and show different points
            # =============================================================================
            self.points_new = copy.deepcopy(self.points)

            ## mousewheel zoom
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
                    
            ## draw points 
            if self.idx > 0:              
                for lm, idx in zip(self.points_new, self.idx_list):
                    cv2.circle(temp_canvas2, lm[0:2], self.point_size, self.point_col, -1)
                    cv2.putText(temp_canvas2,  str(idx), lm[0:2], cv2.FONT_HERSHEY_SIMPLEX, self.label_size, self.label_col,3,cv2.LINE_AA)
            
            ## show typed ID on image
            if self.ID_flag == "query":
                if k > 0 and k != 8 and k != 13 and k != 27:
                    self.ID = self.ID + chr(k)
                elif k == 8:
                    self.ID = self.ID[0:len(self.ID)-1]
                if self.flag_zoom == False: 
                    cv2.putText(temp_canvas2, self.ID, (int(self.image_width/10),int(self.image_height/10)), cv2.FONT_HERSHEY_SIMPLEX, self.label_size+2, black, 3, cv2.LINE_AA)

                    
            ## finish on enter
            if k == 13:
                if self.ID == "":
                    warning_size=int(temp_canvas2.shape[1]/500)
                    w_x = int(temp_canvas2.shape[1]/2)
                    w_y = int(temp_canvas2.shape[0]/2)
                    
                    if self.flag_zoom == False:
                        cv2.putText(temp_canvas2, "No ID entered", (w_x,w_y), cv2.FONT_HERSHEY_SIMPLEX, warning_size , red, warning_size*2, cv2.LINE_AA)
                    else:
                        cv2.putText(temp_canvas2, "No ID entered", (w_x,w_y), cv2.FONT_HERSHEY_SIMPLEX, warning_size, red, warning_size*2, cv2.LINE_AA)
                    
                else:
                    cv2.destroyWindow("phenopype")
                    self.done = True
                    temp_canvas2 = copy.deepcopy(temp_canvas1)
                    break
            
            ## show and reset canvas
            cv2.imshow("phenopype", temp_canvas2)
            temp_canvas2 = copy.deepcopy(temp_canvas1)
            
            ## terminate with ESC
            if k == 27:
                cv2.destroyWindow("phenopype")
                sys.exit("PROCESS TERMINATED")
                break
            
        ## draw full, unzoomed image
        if self.done == True:
            self.drawn = temp_canvas2
            for lm, idx in zip(self.points, self.idx_list):
                cv2.circle(self.drawn, lm[0:2], self.point_size, self.point_col, -1)
                cv2.putText(self.drawn,  str(idx), lm[0:2], cv2.FONT_HERSHEY_SIMPLEX, self.label_size, self.label_col,3,cv2.LINE_AA)
                cv2.putText(self.drawn, self.ID, (int(self.image_width/10),int(self.image_height/10)), cv2.FONT_HERSHEY_SIMPLEX, self.label_size+2, black, 3, cv2.LINE_AA)
            
        ## write points to df
        self.landmarks = self.points      
        if self.idx > 0:
            self.df = pd.DataFrame(data=self.landmarks, columns = ["x","y"], index=list(range(1,self.idx+1)))
            self.df["idx"] = self.idx_list
            self.df["filename"] = self.filename
            self.df["id"] = self.ID
            self.df["scale"] = self.scale
            
        if self.idx > 0:
            self.df["id"] = self.ID

        # =============================================================================
        # add arc-points
        # =============================================================================
        
        if self.draw_line == True:

            ## reset
            print("\nAdd line-points by left clicking, remove by right clicking, finish with enter.")
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("phenopype", self._on_mouse)
            
            temp_canvas1 = copy.deepcopy(self.drawn)
            temp_canvas2 = copy.deepcopy(self.drawn)

            self.done = False 
            self.current = (0, 0) 
            self.current_zoom = []
            self.points = []
            self.points_zoom = []
            self.idx = 0
            self.idx_list = []
            self.flag_zoom = False
            
            while(not self.done):

            # =============================================================================
            # if mousewheel-zoom, update coordinate-space and show different line
            # =============================================================================
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
                                        
                ## draw line and points 
                if self.idx > 0:
                    cv2.polylines(temp_canvas2, np.array([self.points_new]), False, green, 3)
                    cv2.line(temp_canvas2, self.points_new[-1], self.current, blue, 3)     
                for lm, idx in zip(self.points_new, self.idx_list):
                    cv2.circle(temp_canvas2, lm, self.point_size, blue, -1)
                    
                ## show image / finish on enter or esc keystroke
                cv2.imshow("phenopype", temp_canvas2)   
                if cv2.waitKey(50) & 0xff == 13:
                    self.done = True
                    cv2.destroyWindow("phenopype")
                elif cv2.waitKey(50) & 0xff == 27:
                    cv2.destroyWindow("phenopype")
                    
                ## reset canvas
                temp_canvas2 = copy.deepcopy(temp_canvas1)
                
            ## draw full, unzoomed image
            if self.done == True:
                self.drawn = temp_canvas2
                cv2.polylines(self.drawn, np.array([self.points]), False, blue, 3)
                for lm, idx in zip(self.points, self.idx_list):
                    cv2.circle(self.drawn, lm, self.point_size, blue, -1)
                
            ## calculate arc length
            self.arc_points = self.points
            if self.idx > 0:
                self.arc = np.array(self.arc_points)
                self.arc_length = int(cv2.arcLength(self.arc, closed=False))
                self.df["arc_length"] = self.arc_length
            else: 
                self.df["arc_length"] = "NA"
                               
                
        # =============================================================================
        # save and return
        # =============================================================================
            
        if self.draw_line == True:
            self.df = self.df[["filename", "id", "idx", "x","y","scale","arc_length"]]
        else:
            sys.exit("WARNING: No landmarks set!")
            self.df = self.df[["filename", "id", "idx", "x","y","scale"]]
            
        if self.show == True:
            show_img(self.drawn)