#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import numpy.ma as ma
import pandas as pd

#%% methods

def landmark(obj_input, **kwargs):
    """Set landmarks, with option to measure length and enter specimen ID.
    
    Parameters
    ----------
    obj_input : TYPE
        DESCRIPTION.
    scale: num (default: 1)
        pixel to mm-ratio 
    ID: str (default: NA)
        specimen ID; "query" is special flag for user entry
    draw_line: bool (default: False)
        flag to draw arc and measure it's length
    zoom_factor: int (default 5)
        magnification factor on mousewheel use
    show: bool (default: False)
        display the set landmarks 
    point_size: num (default: 1/300 of image diameter)
        size of the landmarks on the image in pixels
    point_col: value (default: red)
        colour of landmark (red, green, blue, black, white)
    label_size: num (1/1500 of image diamter)
        size of the numeric landmark label in pixels
    label_col: value (default: black)
        colour of label (red, green, blue, black, white)    

    Returns
    -------
    .df = pandas data frame with landmarks (and arc-length, if selected)
    .drawn = image array with drawn landmarks (and lines)
    .ID = provided specimen ID 
    """
    ## kwargs
    
    ## load image and contours
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image
        


    out = _image_viewer(image, tool="polygon")


                
    def set_landmarks(self, **kwargs): 


        
        self.ID_flag = kwargs.get("ID","NA")
        if self.ID_flag == "query":
            self.ID = ""
        else:
            self.ID = self.ID_flag
        self.scale = kwargs.get("scale", 1)
        self.zoom_fac = kwargs.get("zoom_factor", 5)
        self.draw_line = kwargs.get("draw_line", False)
        self.show = kwargs.get('show', False)
        
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
            
        else: 
            sys.exit("WARNING: No landmarks set!")

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
            self.df = self.df[["filename", "id", "idx", "x","y","scale"]]
        if self.show == True:
            show_img(self.drawn)
            
        return self.df



def colour(obj_input, **kwargs):

    ## kwargs
    channels = kwargs.get("channels", ["gray"])
    contour_dict = kwargs.get("contours", None)
    contour_df = kwargs.get("df", None)
    
    ## load image and contours
    if obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        if not contour_dict:
            sys.exit("no contours provided")
        if contour_df.__class__.__name__ == "NoneType":
            warnings.warn("no data-frame for contours provided")
            contour_df = pd.DataFrame({"filename":"unknown"}, index=[0]).T
    elif obj_input.__class__.__name__ == "container":
        image = obj_input.image_copy
        contour_dict = obj_input.contours
        contour_df = obj_input.df

    ## create forgeround mask
    image_bin = np.zeros(image.shape[:2], np.uint8)
    for label, contour in contour_dict.items():
        if contour["order"]=="parent":
            image_bin = cv2.fillPoly(image_bin, [contour["coords"]], 255)
        elif contour["order"]=="child":
            image_bin = cv2.fillPoly(image_bin, [contour["coords"]], 0)

    foreground_mask = np.invert(np.array(image_bin, dtype=np.bool))

    ## method
    if "gray" in channels:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_cols = {"gray_mean":"NA",
                    "gray_sd":"NA"}
        contour_df = contour_df.assign(**new_cols)
        for label, contour in contour_dict.items():
            rx,ry,rw,rh = cv2.boundingRect(contour["coords"])
            grayscale =  ma.array(data=image_gray[ry:ry+rh,rx:rx+rw], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            contour_df.loc[contour_df["label"]==label,["gray_mean","gray_sd"]] = np.ma.mean(grayscale), np.ma.std(grayscale)

    if "rgb" in channels:
        contour_df = contour_df.assign(**{"red_mean":"NA",
                           "red_sd":"NA",
                           "green_mean":"NA",
                           "green_sd":"NA",
                           "blue_mean":"NA",
                           "blue_sd":"NA"})
        for label, contour in contour_dict.items():
            rx,ry,rw,rh = cv2.boundingRect(contour["coords"])
            blue =  ma.array(data=image[ry:ry+rh,rx:rx+rw,0], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            green =  ma.array(data=image[ry:ry+rh,rx:rx+rw,1], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            red =  ma.array(data=image[ry:ry+rh,rx:rx+rw,2], mask = foreground_mask[ry:ry+rh,rx:rx+rw])
            contour_df.loc[contour_df["label"]==label,["red_mean","red_sd"]]  = np.ma.mean(red), np.ma.std(red)
            contour_df.loc[contour_df["label"]==label,["green_mean","green_sd"]]  = np.ma.mean(green), np.ma.std(green)
            contour_df.loc[contour_df["label"]==label,["blue_mean","blue_sd"]]  = np.ma.mean(blue), np.ma.std(blue)

    ## return
    if obj_input.__class__.__name__ == "ndarray":
        return contour_df
    elif obj_input.__class__.__name__ == "container":
        obj_input.df = contour_df