import os
import cv2
import numpy as np
import copy
import sys
import pandas as pd 

from phenopype.utils import (show_img)


#%% colours

green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
black = (0,0,0)
white = (255,255,255)



colours = {"red": (0, 0, 255),
 "green": (0, 255, 0), 
 "blue": (255, 0, 0),
 "black":(0,0,0),
 "white":(255,255,255)}


#%% modules
class landmark_maker:
    """Intialize landmarks_maker, loads image.
    
    Parameters
    ----------

    image: str or array
        absolute or relative path to OR numpy array of image 
    scale: num (1)
        pixel to mm-ratio 
    ID: str (NA)
        specimen ID; "query" is special flag for user entry
    point_size: num (1/300 of image diameter)
        size of the landmarks on the image in pixels
    point_col: value (red)
        colour of landmark (red, green, blue, black, white)
    label_size: num (1/1500 of image diamter)
        size of the numeric landmark label in pixels
    label_col: value (black)
        colour of label (red, green, blue, black, white)
    draw_line: bool (False)
        flag to draw arc and measure it's length
    """        
    
    def __init__(self, image, **kwargs):
        
        # initialize # ----------------
        if isinstance(image, str):
            self.image = cv2.imread(image)
            self.filename = os.path.basename(image)
        else:
            self.image = image
            self.filename = kwargs.get("filename","NA")
            
        self.ID = kwargs.get("ID","NA")
        self.scale = kwargs.get("scale", 1)

        self.done = False 
        self.current = (0, 0) 
        self.points = []
        self.idx = 0
        self.idx_list = []

        self.point_size = kwargs.get("point_size", int(((self.image.shape[0]+self.image.shape[1])/2)/300))
        self.point_col = kwargs.get("point_col", red)
        self.label_size = kwargs.get("label_size", int(((self.image.shape[0]+self.image.shape[1])/2)/1500))
        self.label_col = kwargs.get("label_col", black)
        self.draw_line = kwargs.get("draw_line", False)
        

    def _on_mouse(self, event, x, y, buttons, user_param):
        
        if self.done: 
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)          
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.idx += 1
            self.idx_list.append(self.idx)
            print("Landmark #%d with position (x=%d,y=%d) added" % (self.idx, x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points = self.points[:-1]
                self.idx -= 1
                self.idx_list = self.idx_list[:-1]
                print("Landmark #%d with position (x=%d,y=%d) deleted" % (self.idx, x, y))
            else:
                print("No landmarks to delete")
                
                
                
    def draw(self, **kwargs):                   
        
        if not len(self.image.shape)==3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            
        if kwargs.get("zoom", False):
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            (rx,ry,w,h) = cv2.selectROI("phenopype", self.image, fromCenter=False)
            cv2.destroyWindow("phenopype")  
            if any([cv2.waitKey(50) & 0xff == 27, cv2.waitKey(50) & 0xff == 13]):
                cv2.destroyWindow("phenopype")  
            #self.points = [(x, y), (x, y+h), (x+w, y+h), (x+w, y)]
            temp_canvas1 = self.image[ry:ry+h,rx:rx+w]
            temp_canvas2 = temp_canvas1
        else:
            temp_canvas1 = copy.deepcopy(self.image)
            temp_canvas2 = temp_canvas1

        # =============================================================================
        # add landmarks
        # =============================================================================
        
        print("\nAdd landmarks by left clicking, remove by right clicking, finish with enter.")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self._on_mouse)
        
        self.done = False 
        self.current = (0, 0) 
        self.idx = 0
        self.idx_list = []
        self.points = []

        while(not self.done):

            if self.idx > 0:
                for lm, idx in zip(self.points, self.idx_list):
                    cv2.circle(temp_canvas2, lm, self.point_size, self.point_col, -1)
                    cv2.putText(temp_canvas2,  str(idx), lm, cv2.FONT_HERSHEY_SIMPLEX, self.label_size, self.label_col,3,cv2.LINE_AA)
             
            cv2.imshow("phenopype", temp_canvas2)
            
            if cv2.waitKey(50) & 0xff == 13:
                self.done = True
                cv2.destroyWindow("phenopype")
            elif cv2.waitKey(50) & 0xff == 27:
                cv2.destroyWindow("phenopype")
                
            self.drawn = temp_canvas2
            temp_canvas2 = copy.deepcopy(temp_canvas1)
            
        self.landmarks = self.points      
        if self.idx > 0:
            self.df = pd.DataFrame(data=self.landmarks, columns = ["x","y"], index=list(range(1,self.idx+1)))
            self.df["idx"] = self.idx_list
            self.df["filename"] = self.filename
            self.df["id"] = self.ID
            self.df["scale"] = self.scale

        # =============================================================================
        # add arc-points
        # =============================================================================
        
        if self.draw_line == True:
            
            # reset
            print("\nAdd landmarks by left clicking, remove by right clicking, finish with enter.")
            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("phenopype", self._on_mouse)
            
            temp_canvas1 = self.drawn
            temp_canvas2 = self.drawn

            self.done = False 
            self.current = (0, 0) 
            self.idx = 0
            self.idx_list = []
            self.points = []

            
            while(not self.done):
                if self.idx > 0:
                    cv2.polylines(temp_canvas2, np.array([self.points]), False, green, 3)
                    cv2.line(temp_canvas2, self.points[-1], self.current, blue, 3)     
                for lm, idx in zip(self.points, self.idx_list):
                    cv2.circle(temp_canvas2, lm, self.point_size, blue, -1)
                    
                cv2.imshow("phenopype", temp_canvas2)
            
                if cv2.waitKey(50) & 0xff == 13:
                    self.done = True
                    cv2.destroyWindow("phenopype")
                    cv2.destroyWindow("phenopype")
                    
                self.drawn = temp_canvas2
                temp_canvas2 = copy.deepcopy(temp_canvas1)
                
            self.arc_points = self.points
            if self.idx > 0:
                self.arc = np.array(self.arc_points)
                self.arc_length = cv2.arcLength(self.arc, closed=False)
                self.df["arc_length"] = self.arc_length
            else: 
                self.df["arc_length"] = "NA"
                               
                
        # =============================================================================
        # save and return
        # =============================================================================
            
        if self.ID == "query":
            self.ID = input("Enter specimen ID: ")
        
        if self.idx > 0:
            self.df["id"] = self.ID
            
        if self.draw_line == True:
            self.df = self.df[["filename", "id", "idx", "x","y","scale","arc_length"]]
        else:
            self.df = self.df[["filename", "id", "idx", "x","y","scale"]]

        

                    
                
