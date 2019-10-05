import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import copy
import math
import sys
import warnings

import cv2
import datetime

from phenopype.utils import (red, green, blue, white, black)
from phenopype.utils import (blur, exif_date, get_median_grayscale, show_img)


#%% settings

pd.options.display.max_rows = 10

#%% classes

class project_maker: 
    
    """Initialize a phenopype project with a name.
        
        Parameters
        ----------

        name: str (default: "My project, -current date-")
            name of your project     
    """
        
    def __init__(self, **kwargs):
        
        proj_dummy_name = "My project, " + datetime.datetime.today().strftime('%Y-%m-%d')
        self.name = kwargs.get("name", proj_dummy_name)           

                      
    def add_files(self, image_dir, **kwargs):
        
        """Add files to your project from a directory, can look recursively. Optional: specify a search string for filenames or file extensions. 
    
        Parameters
        ----------
    
        image_dir: str 
            path to directory with images                             
        search_mode: str (default: "dir")
            "dir" searches current directory for valid files; "recursive" walks through all subdirectories
        filetypes: list 
            single or multiple string patterns to target files with certain endings
        include: list 
            single or multiple string patterns to target certain files to include - can be used together with exclude
        exclude: list 
            single or multiple string patterns to target certain files to include - can be used together with include

        """
            
        self.in_dir = image_dir
        
        if "save_dir" in kwargs:
            self.save_dir = kwargs.get("save_dir", "local")   
            if self.save_dir == "local":
                self.save_dir = os.path.normpath(self.in_dir) + "_out"
            
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                save_dir_str = str(self.save_dir) + " (created)"
            else:
                save_dir_str = str(self.save_dir)
        else:
            save_dir_str = "No save directory specified."
            
            
        self.mode = kwargs.get("search_mode","dir")                 
        self.filetypes = kwargs.get("filetypes", [])
        self.exclude = kwargs.get("exclude", [])
        self.include = kwargs.get("include", [])
          
        # =============================================================================
        # make file / pathlists
        # =============================================================================       
        
        filepaths1 = []
        filenames1 = []
        if self.mode == "recursive":
            for root, dirs, files in os.walk(self.in_dir):
                for i in os.listdir(root):
                    path = os.path.join(root,i)
                    if os.path.isfile(path):      
                        filepaths1.append(path)
                        filenames1.append(i)
                        
        elif self.mode == "dir":
            for i in os.listdir(self.in_dir):
                path = os.path.join(self.in_dir,i)
                if os.path.isfile(path):      
                    filepaths1.append(path)
                    filenames1.append(i)
                     
        # INCLUDE        
        filepaths2 = []
        filenames2 = []
        if self.include:
            for name, path in zip(filenames1, filepaths1):   
                for inc in self.include:
                    if inc in name:
                        filepaths2.append(path)
                        filenames2.append(name)
        else:
            filepaths2 = filepaths1
            filenames2 = filenames1
    
        # EXCLUDE  
        filepaths3 = []
        filenames3 = []
        if self.exclude:
            for name, path in zip(filenames2, filepaths2):   
                for exc in self.exclude:
                    if exc not in name:
                        filepaths3.append(path)
                        filenames3.append(name)
        else:
            filepaths3 = filepaths2
            filenames3 = filenames2
    
        # FILETYPE        
        filepaths4 = []
        filenames4 = []
        if self.filetypes:
            for name, path in zip(filenames3, filepaths3):   
                for ext in self.filetypes:
                    if name.endswith(ext):
                        filepaths4.append(path)
                        filenames4.append(name)
        else:
            filepaths4 = filepaths3
            filenames4 = filenames3
    
        self.filenames = filenames4
        self.filepaths = filepaths4
    
        # =============================================================================
        # show added files
        # =============================================================================
        
        n_files = len(self.filenames)
        
        print("Search returned " + str(n_files) + " files: \n" + str(self.filenames[0:49]) + "...")
        print("\n")
        print("--------------------------------------------")
        print("Project settings\n================")
        print("Project name: " + self.name + "\nImage directory: " + str(self.in_dir)  + "\nSave directory: " + save_dir_str + "\nSearch mode: " + self.mode + "\nFiletypes: " + str(self.filetypes) + "\nInclude:" + str(self.include) + "\nExclude: " + str(self.exclude) + "\n")
        print("Files\n======")
        if n_files > 10:
            print(str(self.filenames[0:9]) + " ... " + "\n\n Print all filenames or filepaths with _project_name_.filenames / _project_name_.filepaths\n")
        else:
            print(str(self.filenames) + "\n Print filenames or filepaths with _project_name_.filenames / _project_name_.filepaths\n")
        print("--------------------------------------------")
        
                    
class scale_maker:
    
    """Initiate scale maker with an image containing a size-reference card. Make_scale measures 
       the pixel-mm ratio of a reference card, detect_scale can attempt to find the specified card 
       in other images.
        
        Parameters
        ----------
        image: str or array
            absolute or relative path to OR numpy array of image containing the reference card 
    """
        
    def __init__(self, image, **kwargs):
        if isinstance(image, str):
            self.template_image = cv2.imread(image) 
        else:
            self.template_image = image
        if not len(self.template_image.shape)==3:
            self.template_image = cv2.cvtColor(self.template_image, cv2.COLOR_GRAY2BGR)         
                  
    def _on_mouse_scale(self, event, x, y, flags, params):      
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
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.flag_zoom == True:
                x_zoom, y_zoom = self.current_zoom     
                x = x + max(0,x_zoom-self.delta_width)
                y = y + max(0,y_zoom-self.delta_height)
            if len(self.points) < 2:
                self.points.append((x, y))
                self.idx += 1  
                self.idx_list.append(self.idx)
                print("Point %d/2 with position (x=%d,y=%d) added" % (self.idx, x, y))         
            elif len(self.points) == 2:
                print("Finished - no more points to add!")
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points = self.points[:-1]
                self.idx -= 1
                self.idx_list = self.idx_list[:-1]
                print("Point #%d/2 with position (x=%d,y=%d) deleted" % (self.idx, x, y))
            else:
                print("No point to delete")

    def make_scale(self, **kwargs):
        
        """Measure the pixel-to-mm-ratio of a reference card inside an image.
        
        Parameters
        ----------
        
        value: int (default: 10)
            distance to measure between two points
        mode: str (default: "rectangle")
            grab the scale with a polygon or a box
        """
        
        ## kwargs 
        self.zoom_fac = kwargs.get("zoom_factor", 5)                       
        self.flag_mark_scale = kwargs.get("mark_scale", False)
               
        ## initialize with empty lists / parameters
        self.done = False 
        self.current = (0, 0) 
        self.current_zoom = []
        self.points = []
        self.idx = 0
        self.idx_list = []
        self.distance_mm = ""
        self.flag_zoom = False
        
        ## for zoom
        self.image_height = self.template_image.shape[0]
        self.delta_height = int((self.image_height/self.zoom_fac)/2)
        self.image_width = self.template_image.shape[1]
        self.delta_width = int((self.image_width/self.zoom_fac)/2)
        self.image_diag = int((self.image_height + self.image_width)/2)
        
        # =============================================================================
        # measure pixel-to-mm ratio
        # =============================================================================
        
        print("\nMeasure pixel-to-mm-ratio by clicking on two points, and type the distance in mm between them.")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self._on_mouse_scale)
        
        temp_canvas1 = copy.deepcopy(self.template_image)
        temp_canvas2 = copy.deepcopy(self.template_image)
        
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
                    
            # =============================================================================
            # text feedback
            # =============================================================================
                
            ## type distance onto image; accept only digits
            if k > 0 and k != 8 and k != 13 and k != 27:
                if chr(k).isdigit():
                    self.distance_mm = self.distance_mm + chr(k)
            elif k == 8:
                self.distance_mm = self.distance_mm[0:len(self.distance_mm)-1]
              
            ## measure distance in pixel
            if len(self.points) == 2:
                self.distance_px = int(math.sqrt( ((self.points[0][0]-self.points[1][0])**2)+((self.points[0][1]-self.points[1][1])**2)))
            
            ## adjust text size and line width to canvas size            
            text_size = int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/500)
            text_thickness = int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/1000)
            linewidth = int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/500)
                       
            ## text position
            text_mm_y = int(temp_canvas2.shape[0]/10)
            text_mm_x = int(temp_canvas2.shape[1]/50)      
            text_px_y = text_mm_y + text_mm_y
            text_px_x = text_mm_x            
            
            ## draw lines
            if len(self.points) == 1:
                cv2.line(temp_canvas2, self.points_new[-1], self.current, blue, linewidth)
            elif len(self.points) == 2:
                cv2.polylines(temp_canvas2, np.array([self.points_new]), False, green, linewidth)
                
            ## text: length in mm
            if len(self.distance_mm) == 0:
                cv2.putText(temp_canvas2, "Enter distance (mm): ", (text_mm_x, text_mm_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, black, text_thickness, cv2.LINE_AA)
            elif len(self.distance_mm) > 0:
                cv2.putText(temp_canvas2, "Entered distance: " + self.distance_mm + " mm", (text_mm_x, text_mm_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, green, int(text_thickness * 2), cv2.LINE_AA)
            
            ## text: scale measured
            if len(self.points) < 2:
                cv2.putText(temp_canvas2, "Mark scale (2 points): ", (text_px_x, text_px_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, black, text_thickness, cv2.LINE_AA)
            elif len(self.points) == 2:
                cv2.putText(temp_canvas2, "Mark scale: " + str(self.distance_px) + " pixel", (text_px_x, text_px_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, green, int(text_thickness * 2), cv2.LINE_AA)

            ## finish on enter
            if k == 13:
                ## Warning if data needs to be entered
                if len(self.points) < 2:                  
                    cv2.putText(temp_canvas2, "No scale marked!", (int(temp_canvas2.shape[0]/5),int(temp_canvas2.shape[1]/3.5)), cv2.FONT_HERSHEY_SIMPLEX, text_size , red, int(text_thickness * 2), cv2.LINE_AA)
                if len(self.distance_mm) == 0:                  
                    cv2.putText(temp_canvas2, "No distance entered!", (int(temp_canvas2.shape[0]/5),int(temp_canvas2.shape[1]/2.5)), cv2.FONT_HERSHEY_SIMPLEX, text_size , red, int(text_thickness * 2), cv2.LINE_AA)               
                else:
                    cv2.destroyWindow("phenopype")
                    temp_canvas2 = copy.deepcopy(temp_canvas1)
                    self.done = True
                    break
             
            ## show and reset canvas
            cv2.imshow("phenopype", temp_canvas2)
            temp_canvas2 = copy.deepcopy(temp_canvas1) 
            
            ## terminate with ESC
            if k == 27:
                cv2.destroyWindow("phenopype")
                sys.exit("PROCESS TERMINATED")
                break
                
        self.distance_mm = int(self.distance_mm)
        self.scale_ratio = round(self.distance_px/self.distance_mm,3)
        
        print("\n")
        print("------------------------------------------------")
        print("Finished - your scale has %s pixel per mm." % (self.scale_ratio))
        print("------------------------------------------------")
        print("\n")
        
        return self.scale_ratio

    def _on_mouse_scale_template(self, event, x, y, flags, params):      
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
                     
                      
            
        
    def make_scale_template(self, **kwargs):
    
        """Make a template of a reference card inside an image for automatic detection and colour-correction.
        
        Parameters
        ----------
        
        value: int (default: 10)
            distance to measure between two points
        mode: str (default: "rectangle")
            grab the scale with a polygon or a box
        """
        
        ## kwargs 
        self.zoom_fac = kwargs.get("zoom_factor", 5)                       
        self.mode = kwargs.get("mode", "rectangle")                       
        self.show = kwargs.get("show", "True")          

        ## initialize with empty lists / parameters
        self.done = False 
        self.current = (0, 0) 
        self.current_zoom = []
        self.points = []
        self.idx = 0
        self.idx_list = []
        self.distance_mm = ""
        self.flag_zoom = False
        
        ## for zoom
        self.image_height = self.template_image.shape[0]
        self.delta_height = int((self.image_height/self.zoom_fac)/2)
        self.image_width = self.template_image.shape[1]
        self.delta_width = int((self.image_width/self.zoom_fac)/2)
        self.image_diag = int((self.image_height + self.image_width)/2)
            
        print("\nMark outline of scale-template in image, either by drawin polygon outline or by dragging a rectangle over it.")
         
        temp_canvas1 = copy.deepcopy(self.template_image)
        temp_canvas2 = copy.deepcopy(self.template_image)
        
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self._on_mouse_scale_template)

                
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
                    
            ## adjust text size and line width to canvas size            
            text_size = max(int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/1000),2)
            text_thickness = int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/1000)
            linewidth = int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/500)
            
            self.temp_canvas2 = temp_canvas2
                    
            if self.mode == "polygon":
                    
                ## draw lines
                if len(self.points)>0:
                    cv2.line(temp_canvas2, self.points_new[-1], self.current, blue, linewidth)
                    cv2.polylines(temp_canvas2, np.array([self.points_new]), False, green, linewidth)
    
                ## finish on enter
                if k == 13:
                    ## Warning if data needs to be entered
                    if len(self.points) < 3:                  
                        cv2.putText(temp_canvas2, "At least three points needed to draw polygon!", (int(temp_canvas2.shape[0]/2.5),int(temp_canvas2.shape[1]/3.5)), cv2.FONT_HERSHEY_SIMPLEX, text_size, red, int(text_thickness * 2), cv2.LINE_AA)
                    else:
                        cv2.destroyWindow("phenopype")
                        temp_canvas2 = copy.deepcopy(temp_canvas1)
                        self.done = True
                        break
                               
            elif self.mode == "rectangle":
                                   
                ## draw rectangle
                if len(self.points)==1:
                    cv2.rectangle(temp_canvas2, self.points_new[0], self.current, blue, linewidth)
                if len(self.points)==2:
                    cv2.rectangle(temp_canvas2, self.points_new[0], self.points_new[1], blue, linewidth)
                
                ## finish on enter
                if k == 13:
                    ## Warning if data needs to be entered
                    if len(self.points) < 2:                  
                        cv2.putText(temp_canvas2, "At least two points needed to draw rectangle!", (int(temp_canvas2.shape[0]/2.5),int(temp_canvas2.shape[1]/3.5)), cv2.FONT_HERSHEY_SIMPLEX, text_size, red, int(text_thickness * 2), cv2.LINE_AA)
                    else:                      
#                        (x1,y1) = self.points[0]
#                        (x2,y2) = self.points[1]
#                        self.points_poly = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
                        
                        cv2.destroyWindow("phenopype")
                        temp_canvas2 = copy.deepcopy(temp_canvas1)
                        self.done = True
                        break
                
            ## show and reset canvas
            cv2.imshow("phenopype", temp_canvas2)
            temp_canvas2 = copy.deepcopy(temp_canvas1) 
            
            ## terminate with ESC
            if k == 27:
                cv2.destroyWindow("phenopype")
                sys.exit("PROCESS TERMINATED")
                break
            
        # create bounding box and cut template
        x,y,w,h = cv2.boundingRect(np.array(self.points))
        self.points_poly = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
        self.template = self.template_image[y:y+h,x:x+w]

        # create colour mask to show template 
        self.colour_mask = np.zeros(self.template_image.shape, np.uint8)
        self.colour_mask[:,:,1] = 255 # all area green
        cv2.fillPoly(self.colour_mask, np.array([self.points_poly]), red) # red = excluded area
        self.overlay = cv2.addWeighted(copy.deepcopy(self.template_image), .7, self.colour_mask, 0.3, 0) # combine
        if self.mode=="polygon":
            points_drawing = self.points 
            points_drawing.append(self.points[0])
            cv2.polylines(self.overlay, np.array([points_drawing]), False, green, linewidth)
        if self.mode=="rectangle":
            cv2.rectangle(self.overlay, self.points[0], self.points[1], green, linewidth)

            
        if self.show == True:
            show_img(self.overlay)

        print("\n")
        print("------------------------------------------------")
        print("Finished - scale template completed")
        print("------------------------------------------------")
        print("\n")
            
#        
#        # create mask for SIFT
#        self.mask_original_template = np.zeros(image.shape[0:2], np.uint8)
#        cv2.fillPoly(self.mask_original_template, [np.array(self.points_step1)], white) 
#        self.mask_original_template = self.mask_original_template[ry:ry+h,rx:rx+w]
#        
#        # zoom into drawn scale outline for better visibility
#        if zoom==True:
#            temp_canvas_1 = image[ry:ry+h,rx:rx+w]
#            self.done_step1 = True

        # =============================================================================
        # Step 2 - measure scale length
        # =============================================================================
        
#        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
#        temp_canvas_2 = copy.deepcopy(temp_canvas_1)
#        cv2.setMouseCallback("phenopype", self._on_mouse_step2)
#        
#        while(not self.done_step2):
#            if (len(self.points_step2) > 0):
#                cv2.polylines(temp_canvas_2, np.array([self.points_step2]), False, green, 2)
#                cv2.line(temp_canvas_2, self.points_step2[-1], self.current, blue, 2)
#            cv2.imshow("phenopype", temp_canvas_2)
#            temp_canvas_2 = copy.deepcopy(temp_canvas_1)
#            if any([cv2.waitKey(50) & 0xff == 27, cv2.waitKey(50) & 0xff == 13]):
#                cv2.destroyWindow("phenopype")
#                break    
#            

#        if kwargs.get('show', True):
#            cv2.polylines(temp_canvas_2, np.array([self.points_step2]), False, red, 4)    
#            cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
#            cv2.imshow("phenopype", temp_canvas_2)    
#            if any([cv2.waitKey(0) & 0xff == 27, cv2.waitKey(0) & 0xff == 13]):
#                cv2.destroyWindow("phenopype")
#              
#        self.image_zoomed = temp_canvas_2

#        # SCALE
#        self.measured = self.scale_px/value
#        self.current = self.measured
#
#        # REFERENCE
#        (x,y),radius = cv2.minEnclosingCircle(np.array(self.points_step1))
#        self.ref = (radius * 2)
#        
#        # MASK
#        zeros = np.zeros(image.shape[0:2], np.uint8)
#        self.mask = cv2.fillPoly(zeros, [np.array(self.points_step1, dtype=np.int32)], white)
#        
#        # create mask object to use in object finder
#        include = False
#        self.mask_obj = np.array(self.mask, dtype=bool), "scale", include
    
    
    def detect(self, image, **kwargs):
        """Find scale from a defined template inside an image and update pixel ratio. Feature detection is run by the AKAZE algorithm (http://www.bmva.org/bmvc/2013/Papers/paper0013/abstract0013.pdf).  
        
        Parameters
        -----------
        image: str or array
            absolute or relative path to OR numpy array of image containing the scale 
        show: bool (optional, default: False)
            show result of scale detection procedure on current image   
        resize: num (optional, default: 1)
            resize image to speed up detection process (WARNING: too low values may result in poor detection results or even crashes)
        """
        
        # =============================================================================
        # INITIALIZE
        # =============================================================================
        
        if isinstance(image, str):
            self.image_target = cv2.imread(image)
        else:
            self.image_target = image

        image_target = self.image_target 
        image_original = self.image_original_template
        
        show = kwargs.get('show', False)
        min_matches = kwargs.get('min_matches', 10)
        
        # image diameter bigger than 2000 px
        if (image_target.shape[0] + image_target.shape[1])/2 > 2000:
            factor = kwargs.get('resize', 0.5)
        else:
            factor = kwargs.get('resize', 1)
        image_target = cv2.resize(image_target, (0,0), fx=1*factor, fy=1*factor) 
        
        if not len(image_target.shape)==3:
            image_target = cv2.cvtColor(image_target, cv2.COLOR_GRAY2BGR)
            
    
        # =============================================================================
        # SIFT detector
        # =============================================================================
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp1, des1 = sift.detectAndCompute(img1,self.mask_original_template)
        # kp2, des2 = sift.detectAndCompute(img2,None)
         
        # =============================================================================
        # ORB detector
        # =============================================================================
#        orb = cv2.ORB_create()
#        kp1, des1 = orb.detectAndCompute(img1,self.mask_original_template)
#        kp2, des2 = orb.detectAndCompute(img2,None)
#        des1 = np.asarray(des1, np.float32)
#       des2 = np.asarray(des2, np.float32)
        
#        FLANN_INDEX_KDTREE = 0
#        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#        search_params = dict(checks = 50)
#        flann = cv2.FlannBasedMatcher(index_params, search_params)
#        matches = flann.knnMatch(des1,des2,k=2)
        
        # =============================================================================
        # AKAZE detector
        # =============================================================================     
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(image_original,self.mask_original_template)
        kp2, des2 = akaze.detectAndCompute(image_target,None)       
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        matches = matcher.knnMatch(des1, des2, 2)

        # keep only good matches
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        self.nkp = len(good)
        
        # find and transpose coordinates of matches
        if self.nkp >= min_matches:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            ret, contours, hierarchy = cv2.findContours(self.mask_original_template,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_TC89_L1)
            box = contours[0].astype(np.float32)

            rect  = cv2.perspectiveTransform(box,M).astype(np.int32)
            image_target = cv2.polylines(image_target,[rect],True,red,5, cv2.LINE_AA)
            
            # =============================================================================
            # compare scale to original, and return adjusted ratios
            # =============================================================================
            
            if show == True:
                cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
                cv2.imshow("phenopype", image_target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if kwargs.get("convert",True) == True:
                rect = rect/factor
                
            # SCALE
            rect = np.array(rect, dtype=np.int32)
            (x,y),radius = cv2.minEnclosingCircle(rect)
            self.current = round(self.measured * ((radius * 2)/self.ref),1)
            
            # MASK
            zeros = np.zeros(self.image_target.shape[0:2], np.uint8)
            mask_bin = cv2.fillPoly(zeros, [np.array(rect)], white)       
            self.mask = np.array(mask_bin, dtype=bool)

            # TARGET SNIPPET
            (rx,ry,w,h) = cv2.boundingRect(rect)
            self.image_found = self.image_target[ry:ry+h,rx:rx+w]

            
            print("\n")
            print("--------------------------------------")
            print("Scale found with %d keypoint matches" % self.nkp)
            print("--------------------------------------")
            print("\n")

            return (self.mask, "scale", False), self.current       
        
        else:
            print("\n")
            print("----------------------------------------------")
            print("Scale not found - only %d/%d keypoint matches" % (self.nkp, min_matches))
            print("----------------------------------------------")
            print("\n")
            
            return "no current scale", "no scale mask"
        
class mask_maker:
    """Intialize mask maker, loads image.
    
    Parameters
    ----------

    image: str or array
        absolute or relative path to OR numpy array of image 
    """        
    def __init__(self, image):
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
        
        self.overlay = np.zeros(self.image.shape, np.uint8) # make overlay
        self.overlay[:,:,2] = 200 # start with all-red overlay
        
    def _on_mouse(self, event, x, y, buttons, user_param):
        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.idx += 1
            print("Adding point #%d with position(%d,%d) to arena" % (self.idx, x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points = self.points[:-1]
                self.idx -= 1
                print("Removing point #%d with position(%d,%d) from arena" % (self.idx, x, y))
            else:
                print("No points to delete")
                
    def draw_mask(self, **kwargs):
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
    
    
class object_finder:
    """Initialize object finder class, loads image.
        
        Parameters
        ----------

        image: str or array
            absolute or relative path to OR numpy array of image containing the objects 

    """
    def __init__(self, image):

        if isinstance(image, str):
            self.image = cv2.imread(image)
            self.filename = os.path.basename(image)
            try:
                self.date_taken = exif_date(image)
            except:
                self.date_taken = "NA"       
        else: 
            self.image = image
            self.date_taken = "NA"      
            
    def find_objects(self, **kwargs):    
        """Method in object finder class: find objects in colour or grayscale images using thresholding
        
        Parameters
        ----------
        method: list (default: ["otsu"])
            determines the type of thresholding: 
                - "binary" needs an interger for the threshold value (default: 127), 
                - "adaptive" needs odd integer for blocksize (default: 99) and constant to be subtracted (default 1) 
                - for more info see https://docs.opencv.org/3.4.4/d7/d4d/tutorial_py_thresholding.html
        operations: list (default: ["diameter", "area"])
            determines the type of operations to be performed on the detected objects:
                - "diameter" of the bounding circle of our object
                - "area" within the contour of our object
                - "grayscale" mean and standard deviation of grayscale pixel values inside the object contours
                - "bgr" mean and standard deviation of blue, green and red pixel values inside the object contours
                - "skeletonize" attempts to transform object into a skeleton form using the technique of Zhang-Suen. WARNING: can be slow for large objects
        scale: num (1)
            pixel to mm-ratio 
        mode: str (default: "multiple")
            detect all, or only ["single"] largest object or multiple 
        mask: list
            phenoype mask-objects (lists of boolean mask, label, and include-argument) to include or exclude an area from the procedure
        show: bool (default: True)
            display the detection results
        blur1: int
            first pass blurring kernel size (before thresholding)
        blur2: list
            second pass blurring kernel size (after thresholding) and binary thresholding value (default 127)   
        min_diam: int
            minimum diameter (longest distance in contour) in pixels for objects to be included (default: 0)
        min_area: int
            minimum contour area in pixels for objects to be included (default: 0)
        corr_factor: int
            factor (in px) to add to (positive int) or subtract from (negative int) object (default: 0)
        resize: in (0.1-1)
            resize image to speed up detection process - usually not recommended
        gray_value_ref: int (0-255)
            reference gray scale value to adjust the given picture's histogram to
            
        """

        resize = kwargs.get("resize", 1)
        show = kwargs.get('show', True)
        mask_objects = kwargs.get('mask', [])

        self.image = cv2.resize(self.image, (0,0), fx=1*resize, fy=1*resize) 
        
        if not "scale" in kwargs:
            print("Warning - no scale specified")
        self.scale = kwargs.get("scale", 1)
        
        self.mode =  kwargs.get('mode', "multiple")
        self.operations =  ["area", "diameter"]
        self.operations = self.operations + kwargs.get('operations', [])
        self.date_analyzed = str(datetime.datetime.now())[:19]


        # APPLY GRAY-CORRECTION FACTOR TO GRAYSCALE IMAGE AND ROI
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        if 'gray_value_ref' in kwargs:
            if resize < 1:
                ret = get_median_grayscale(source=self.gray)
            else: 
                ret = get_median_grayscale(source=self.gray,  resize=0.25)
            ref = kwargs.get('gray_value_ref', ret)
            self.gray_corr_factor = int(ref - ret)
            self.gray_corrected = np.array(copy.deepcopy(self.gray) + self.gray_corr_factor, dtype="uint8")
            self.image_processed = self.gray_corrected
        else:
             self.image_processed = copy.deepcopy(self.image)     
             
       
        #self.image_processed = cv2.cvtColor(self.image_processed,cv2.COLOR_GRAY2BGR)
        
        
        # =============================================================================
        # BLUR1 > THRESHOLDING > MORPHOLOGY > BLUR2
        # =============================================================================
        
        # BLUR 1ST PASS
        if "blur1" in kwargs:
            blur_kernel = kwargs.get("blur1", 1)
            self.blurred = blur(self.gray, blur_kernel)
        else:
            self.blurred = self.gray
            
        # THRESHOLDING   
        method = kwargs.get('method', ["otsu"])
        if isinstance(method, list):
            thresholding = method[0]
            self.thresholding = method[0]
        else:
            sys.exit("ERROR: List type expected for \"thresholding\"")
            
        if thresholding == "otsu":
            ret, self.thresh = cv2.threshold(self.blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif thresholding == "adaptive":
            if not method[1]:
                method[1] = 99
            if not method[2]:
                method[2] = 1
            self.thresh = cv2.adaptiveThreshold(self.blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,method[1], method[2])
        elif thresholding == "binary":
            if not method[1]:
                method[1] = 127
            ret, self.thresh = cv2.threshold(self.blurred,method[1], 255,cv2.THRESH_BINARY_INV)  
        self.morph = copy.deepcopy(self.thresh)
                   
        # BLUR 2ND PASS
        if "blur2" in kwargs:
            blur_kernel, thresh_val = kwargs.get("blur2")
            self.morph = blur(self.morph, blur_kernel)
            ret, self.morph = cv2.threshold(self.morph, thresh_val, 255,cv2.THRESH_BINARY)

            
        # BORDER CORRECTION FACTOR
        if "corr_factor" in kwargs:
            corr_factor = kwargs.get("corr_factor")
            
            size = abs(corr_factor[1])
            iterations = corr_factor[2]
            
            if corr_factor[0] == "cross":
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(size,size))
            elif corr_factor[0] == "ellipse":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
                       
            corr_factor = kwargs.get('corr_factor', 0)
            if corr_factor[1] < 0:
                self.morph = cv2.erode(self.morph, kernel, iterations = iterations)
            if corr_factor[1] > 0:
                self.morph = cv2.dilate(self.morph, kernel, iterations = iterations)

    
#        # APPLY ARENA MASK
#        if "exclude" in kwargs:
#            self.mask = sum(kwargs.get('exclude'))
#            self.mask = cv2.resize(self.mask, (0,0), fx=1*resize, fy=1*resize) 
#            self.morph = cv2.subtract(self.morph,self.mask)
#            self.morph[self.morph==1] = 0
#            

        
        # =============================================================================
        # masking
        # =============================================================================
        
        if len(mask_objects)>0:
                
            mask_dummy1 = np.zeros(self.morph.shape, dtype=np.int8)
            
            for obj in mask_objects:
                mask, label, include = obj
                if include == True:
                    mask_dummy2 = np.zeros(self.morph.shape, dtype=np.int8)
                    mask_dummy2[mask] = 1
                    mask_dummy1 = np.add(mask_dummy1, mask_dummy2)
                if include == False:
                    mask_dummy2 = np.zeros(self.morph.shape, dtype=np.int8)
                    mask_dummy2[mask] = -100
                    mask_dummy1 = np.add(mask_dummy1, mask_dummy2)

            self.morph[mask_dummy1<=0]=0


        # =============================================================================
        # MULTI-MODE
        # =============================================================================

        df_list = []
        df_column_names = []
        label = kwargs.get("label", True)

        if self.mode == "multiple":
            idx = 0
            idx_noise = 0
            length_idx = 0
            area_idx = 0       
           
#            self.roi_bgr_list = []
#            self.roi_gray_list = []
#            self.roi_mask_list = []
            
            ret, self.contours, hierarchy = cv2.findContours(self.morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
            
            # LOOP THROUGH CONTOURS 
            if self.contours:
                for cnt in self.contours:
                    if len(cnt) > 5:
                        
                        # SIZE / AREA CHECK
                        (x,y),radius = cv2.minEnclosingCircle(cnt)
                        x, y= int(x), int(y)
                        diameter = int(radius * 2)
                        area = int(cv2.contourArea(cnt))
                        cont = True
                        if not diameter > kwargs.get('min_diam', 0):
                            length_idx += 1
                            cont = False
                        if not area > kwargs.get('min_area', 0):
                            area_idx += 1
                            cont=False
                        if not cont == False:
                            idx += 1
                            rx,ry,rw,rh = cv2.boundingRect(cnt)
                            
                            cnt_list = []
                            cnt_list = cnt_list + ([self.filename, self.date_taken, self.date_analyzed, idx, x, y, self.scale])
                            df_column_names = []          
                            df_column_names = df_column_names + ["filename","date_taken", "date_analyzed", "idx", "x", "y", "scale"]    
                            
                            # =============================================================================
                            # OPERATIONS TO PERFORM ON MASKED IMAGE                            
                            # =============================================================================

                            if any("diameter" in o for o in self.operations):
                                cnt_list.append(diameter)
                                df_column_names.append("diameter")

                            if any("skeletonize" in o for o in self.operations):
                                cnt_mask = np.zeros(self.gray.shape, np.uint8)
                                img_cnt = cv2.drawContours(cnt_mask, [cnt], 0, white,-1)
                                skeleton=cv2.ximgproc.thinning(img_cnt)                                
                                skel_ret, skel_contour, skel_hierarchy = cv2.findContours(skeleton,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)  
                                skel_contour = skel_contour[0]
                                skeleton_dist = int(cv2.arcLength(skel_contour, closed=False)/2/self.scale)
                                cnt_list.append(skeleton_dist)
                                df_column_names.append("skeleton_dist")                               

                            if any("area" in o for o in self.operations):
                                cnt_list.append(area)                                
                                df_column_names.append("area")
        
                            if any("grayscale" in o for o in self.operations):
                                grayscale =  ma.array(data=self.gray[ry:ry+rh,rx:rx+rw], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                                grayscale_mean = int(np.mean(grayscale)) 
                                grayscale_sd = int(np.std(grayscale)) 
                                cnt_list = cnt_list + [grayscale_mean, grayscale_sd]
                                df_column_names = df_column_names + ["grayscale_mean","grayscale_sd"]
        
                            if any("bgr" in o for o in self.operations):
                                b =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,0], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                                g =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,1], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                                r =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,2], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                                bgr_mean = (int(np.mean(b)),int(np.mean(g)),int(np.mean(r))) # mean grayscale value
                                bgr_sd = (int(np.std(b)),int(np.std(g)),int(np.std(r))) # mean grayscale value
                                cnt_list = cnt_list + [bgr_mean, bgr_sd]
                                df_column_names = df_column_names + ["bgr_mean","bgr_sd"]
                                
                            df_list.append(cnt_list)    
                              
#                            self.roi_bgr_list.append(self.image[ry:ry+rh,rx:rx+rw,])
#                            self.roi_gray_list.append(self.gray[ry:ry+rh,rx:rx+rw,])
#                            self.roi_mask_list.append(self.thresh[ry:ry+rh,rx:rx+rw])

                            # DRAW TO ROI
                            q=kwargs.get("roi_size",300)/2
                            if label==True:
                                cv2.putText(self.image_processed,  str(idx) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),5,cv2.LINE_AA)
                            cv2.drawContours(self.image_processed, [cnt], 0, blue, int(5 * resize))
                            if any("skeletonize" in o for o in self.operations):                    
                                cv2.drawContours(self.image_processed, [skel_contour], 0, green, 2)

                    else:
                        idx_noise += 1
            else: 
                print("No objects found - change parameters?")
            
                        
        # =============================================================================
        # SINGLE-MODE
        # =============================================================================
                
        elif self.mode =="single":
            ret, self.contours, hierarchy = cv2.findContours(self.morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
            
            # LOOP THROUGH CONTOURS AND PICK LARGEST
            if self.contours:
                areas = [cv2.contourArea(cnt) for cnt in self.contours]                
                cnt = self.contours[np.argmax(areas)]
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                x, y= int(x), int(y)
                diameter = int(radius * 2)
                area = int(cv2.contourArea(cnt))
                if diameter > kwargs.get('min_diam', 0) and area > kwargs.get('min_area', 0):
                    # return contour
                    self.contour = cnt
                    idx = 1
                    rx,ry,rw,rh = cv2.boundingRect(cnt)
                    
                    df_list = df_list + [self.filename, self.date_taken, self.date_analyzed, idx, x, y, self.scale]
                    df_column_names = df_column_names + ["filename","date_taken", "date_analyzed", "idx", "x", "y", "scale"]    
                    
                    # =============================================================================
                    # OPERATIONS TO PERFORM ON MASKED IMAGE                            
                    # =============================================================================

                    if any("diameter" in o for o in self.operations):
                        df_list.append(diameter)
                        df_column_names.append("diameter")
                        
                    if any("skeletonize" in o for o in self.operations):
                        cnt_mask = np.zeros(self.gray.shape, np.uint8)
                        img_cnt = cv2.drawContours(cnt_mask, [cnt], 0, white,-1)
                        skeleton=cv2.ximgproc.thinning(img_cnt)                                
                        skel_ret, skel_contour, skel_hierarchy = cv2.findContours(skeleton,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)  
                        skel_contour = skel_contour[0]
                        skeleton_dist = int(cv2.arcLength(skel_contour, closed=False)/2/self.scale)
                        df_list.append(skeleton_dist)
                        df_column_names.append("skeleton_dist")                
                
                    if any("area" in o for o in self.operations):
                        df_list.append(area)                                
                        df_column_names.append("area")

                    if any("grayscale" in o for o in self.operations):
                        grayscale =  ma.array(data=self.gray[ry:ry+rh,rx:rx+rw], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                        grayscale_mean = int(np.mean(grayscale)) 
                        grayscale_sd = int(np.std(grayscale)) 
                        df_list = df_list + [grayscale_mean, grayscale_sd]
                        df_column_names = df_column_names + ["grayscale_mean","grayscale_sd"]

                    if any("bgr" in o for o in self.operations):
                        b =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,0], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                        g =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,1], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                        r =  ma.array(data=self.image[ry:ry+rh,rx:rx+rw,2], mask = np.logical_not(self.thresh[ry:ry+rh,rx:rx+rw]))
                        bgr_mean = (int(np.mean(b)),int(np.mean(g)),int(np.mean(r))) # mean grayscale value
                        bgr_sd = (int(np.std(b)),int(np.std(g)),int(np.std(r))) # mean grayscale value
                        df_list = df_list + [bgr_mean, bgr_sd]
                        df_column_names = df_column_names + ["bgr_mean","bgr_sd"]

                    # DRAW TO ROI
                    if "roi_size" in kwargs:
                        q=kwargs.get("roi_size",300)/2
                        cv2.rectangle(self.image_processed,(int(max(0,x-q)),int(max(0, y-q))),(int(min(self.image.shape[1],x+q)),int(min(self.image.shape[0],y+q))),red,8)
                    cv2.drawContours(self.image_processed, [cnt], 0, red, int(5 * resize))
                    if any("skeletonize" in o for o in self.operations):                    
                        cv2.drawContours(self.image_processed, [skel_contour], 0, green, 2)

                else: 
                    print("Object not bigger than minimum diameter or area")
            else: 
                print("No objects found - change parameters?")

        # =============================================================================
        # RETURN DF AND IMAGE
        # =============================================================================    
        
        # CREATE DF
        
        if any(isinstance(el, list) for el in df_list):
            self.df = pd.DataFrame(data=df_list, columns = df_column_names)
        elif len(df_list)>0:
            self.df = pd.DataFrame(data=[df_list], columns = df_column_names)
            self.df.set_index('filename', drop=True, inplace=True)
            self.df.insert(3, "resize_factor", resize)
            if hasattr(self,'gray_corr_factor'):
                self.df.insert(3, "gray_corr_factor", self.gray_corr_factor)
        else: 
            self.df = pd.DataFrame(data=[["NA"] * len(df_column_names)], columns = df_column_names)
            print("No objects found with these settings!")
            
        
        
        
    
        # SHOW IMAGE
        if hasattr(self,'mask'):
            boo = np.array(self.mask, dtype=bool)
            #self.image_processed = copy.deepcopy(self.image)
            self.image_processed[boo,2] = 255

        if show == True:
            show_img(self.image_processed)
            

        # =============================================================================
        # FEEDBACK + RETURN
        # =============================================================================
        
        if len(df_list)>0:
                    
            self.df_short = self.df[["idx", "diameter", "area"]]
            self.df_short.set_index("idx", inplace=True)
            
            all_pts = len(self.contours)
            good_pts = len(self.df)
            
            if self.mode == "multiple":
                
                print(self.df_short)
                print("\n")
                print("----------------------------------------------------------------")
                print("Found " + str(all_pts) + " objects in " + self.filename + ":")
                print("  ==> %d are valid objects" % good_pts)
                if not idx_noise == 0:
                    print("    - %d are noise" % idx_noise)
                if not length_idx == 0:
                    print("    - %d are not bigger than minimum diameter" % length_idx)
                if not area_idx ==0:
                    print("    - %d are not bigger than minimum area" % area_idx)
                print("----------------------------------------------------------------")
                
            else:
                print("----------------------------------------------------------------")
                print("Found following object in " + self.filename + ":")
                print("----------------------------------------------------------------")
                print(self.df_short)
    
            return self.df

