import copy
import cv2
import datetime
import math
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
import sys
import shutil
import stat
import yaml

from shutil import copyfile

from phenopype.utils import (red, green, blue, white, black)
from phenopype.utils import (blur, exif_date, get_median_grayscale, show_img)
from phenopype.utils_lowlevel import _image_viewer

#%% settings

pd.options.display.max_rows = 10

def del_rw(action, name, exc):
    os.chmod(name, stat.S_IWRITE)
    os.remove(name)

#%% classes
    
class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)
               

class project: 
    """
    Initialize a phenopype project with a name.
    
    Parameters
    ----------

    root_dir: str (default: "CurrentWorkingDir + "CurrentDate_phenopype")
        root directory of the project
    name: str (default: "CurrentDate_phenopype")
        name of your project
    """
    
    def __init__(self, **kwargs):

        flag_overwrite = kwargs.get("overwrite", False)
        flag_delete = kwargs.get("delete", False)
        flag_create = False
        
        if "root_dir" in kwargs:
            root_dir = kwargs.get("root_dir")
            if os.path.basename(root_dir) == "":
                name = kwargs.get("name", datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + "_phenopype_project")
                root_dir = os.path.join(root_dir, name)
            else:
                name = os.path.basename(root_dir)
        elif "name" in kwargs:
            name = kwargs.get("name")
            root_dir = os.path.join(os.getcwd(), name)
        else:
            name = datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + "_phenopype_project"
            root_dir = os.path.join(os.getcwd(), name)

        print("\n")
        print("--------------------------------------------")
        print("phenopype will create a new project named \"" + name + "\". The full path of the project's root directory will be:\n")
        print(root_dir + ".phenopype")
        create = input("Proceed? (y/n)\n")
        if create=="y" or create == "yes":
            if os.path.isdir(root_dir + ".phenopype"):
                if flag_overwrite == True:
                    if flag_delete == True:
                        shutil.rmtree(root_dir + ".phenopype", ignore_errors=True, onerror=del_rw) 
                        print("\n\"" + root_dir + ".phenopype\" and its content deleted")
                    flag_create = True
                else:
                    overwrite = input("Warning - project already exists - overwrite? (y/n)")
                    if overwrite=="y" or overwrite == "yes":
                        if flag_delete == True:
                            shutil.rmtree(root_dir + ".phenopype", ignore_errors=True, onerror=del_rw) 
                            print("\n\"" + root_dir + ".phenopype\" and its content deleted")
                        flag_create = True
            else:
                    flag_create = True
                    
        if flag_create == True:
            self.name = name
            self.root_dir = os.path.abspath(root_dir) + ".phenopype"
            self.data_dir = os.path.join(self.root_dir, "data")
            self.config_filepath = os.path.join(self.root_dir, self.name + '.phenopype.yaml')
            self.filepaths_original = []
            self.filepaths = []
            self.filenames = []
            self.data = {}
            
            os.mkdir(self.root_dir)
                                    
            config = {"date_created": datetime.datetime.today().strftime('%Y%m%d_%H%M%S'),
                      "date_changed": datetime.datetime.today().strftime('%Y%m%d_%H%M%S'),
                      "filepaths_original": self.filepaths_original}
            with open(self.config_filepath, 'w') as config_file:
                yaml.dump(config, config_file, default_flow_style=False) 
            print("\n\"" + name + "\" created at \"" + root_dir + ".phenopype\"")
            print("--------------------------------------------")
        else: 
            print("\n\"" + name + "\" not created!")
            print("--------------------------------------------")    

                      
    def add_files(self, image_dir, **kwargs):
        """Add files to your project from a directory, can look recursively. 
        Optional: specify a search string for filenames or file extensions. 
    
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
        
        ## kwargs                      
        search_mode = kwargs.get("search_mode","dir")                 
        file_endings = kwargs.get("filetypes", [])
        exclude_args = kwargs.get("exclude", [])
        include_args = kwargs.get("include", [])
        unique_by = kwargs.get("unique_by", "filepaths")
        
        ## dummy filepaths for refinement
        filepaths1, filepaths2, filepaths3, filepaths4 = [],[],[],[]

        ## find files 
        if search_mode == "recursive":
            for root, dirs, files in os.walk(image_dir):
                for file in os.listdir(root):
                    filepath = os.path.join(root,file)
                    if os.path.isfile(filepath):
                        filepaths1.append(filepath)
        elif search_mode == "dir":
            for file in os.listdir(image_dir):
                filepath = os.path.join(image_dir,file)
                if os.path.isfile(filepath):   
                    filepaths1.append(filepath)
                    
        ## file endings
        if len(file_endings)>0:
            for filepath in filepaths1:
                if filepath.endswith(tuple(file_endings)):
                    filepaths2.append(filepath)
        elif len(file_endings)==0:
            filepaths2 = filepaths1
            
        ## include
        if len(include_args)>0:
            for filepath in filepaths2:   
                if any(inc in os.path.basename(filepath) for inc in include_args):
                    filepaths3.append(filepath)
        else:
            filepaths3 = filepaths2
            
        ## exclude
        if len(exclude_args)>0:
            for filepath in filepaths3:   
                if not any(exc in os.path.basename(filepath) for exc in exclude_args):
                    filepaths4.append(filepath)
        else:
            filepaths4 = filepaths3
                
        
        ## save to object
        filepaths = filepaths4
        filenames = []
        for filepath in filepaths:
            filenames.append(os.path.basename(filepath))       
        
        ## allow unique filenames filepath or by filename only
        if unique_by=="filepaths":
            self.filepaths_not_added = []
            for filename, filepath in zip(filenames, filepaths):
                if not filepath in self.filepaths_original:
                    self.filepaths_original.append(filepath)
                    self.filenames.append(filename)
                else:
                    self.filepaths_not_added.append(filepath)
        elif unique_by=="filenames":
            self.filepaths_not_added = []
            for filename, filepath in zip(filenames, filepaths):
                if not filename in self.filenames:
                    self.filepaths_original.append(filepath)
                    self.filenames.append(filename)
                else:
                    self.filepaths_not_added.append(filepath)
                    
        ## build phenopype folder structure
        ## data folder in root
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        ## loop through files
        for filepath_original in self.filepaths_original:
            
            ## flatten folder structure
            subfolder_prefix = os.path.dirname(os.path.relpath(filepath_original,image_dir)).replace("\\","__")
            filename = subfolder_prefix + "____" + os.path.basename(filepath_original)
            filepath = os.path.join(self.root_dir,"data",os.path.splitext(filename)[0])
            if not os.path.isdir(filepath):
                
                ## make image-specific directories
                os.mkdir(filepath)
                
                ## copy and rename raw-file
                copyfile(filepath_original, os.path.join(filepath,"raw.png"))
                
                ## specify attributes
                attributes = {"date_created": datetime.datetime.today().strftime('%Y%m%d_%H%M%S'),
                      "date_changed": datetime.datetime.today().strftime('%Y%m%d_%H%M%S'),
                      "filepaths_original": filepath_original}
                with open(filepath + "\\settings.phenopype.yaml", 'w') as config_file:
                    yaml.dump(attributes, config_file, default_flow_style=False)                    
                    
                ## add to project object
                self.filepaths.append(filepath)
                self.data[os.path.splitext(filename)[0]] = obj()
                
                ## feedback
                print(filename + " added")
            else:
                print(filename + " already exists")

        
        def raw(self):
            
        
        
        ## update config file
        with open(self.config_filepath, 'r') as config_file:
            config = yaml.safe_load(config_file)
        config["date_changed"] = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        config["filepaths_original"] = self.filepaths_original
        with open(self.config_filepath, 'w') as config_file:
            config = yaml.dump(config, config_file, default_flow_style=False) 
        
                    
class scale:
    
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
 
                   
                  
    def _on_mouse_measure_scale(self, event, x, y, flags, params):      
        """- Internal reference - don't call this directly -  
        
        Mouse events for "measure_scale" function.
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



    def _on_mouse_make_scale_template(self, event, x, y, flags, params):      
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



    def _equalize(self, **kwargs):
        """- Internal reference - don't call this directly -  
        
        Histogram equalization via interpolation, upscales the results from the detected reference card to the entire image.
        May become a standalone function at some point in the future. THIS STRONGLY DEPENDS ON THE QUALITY OF YOUR TEMPLATE.
        Mostly inspired by this SO question: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
        More theory here: https://docs.opencv.org/master/d4/d1b/tutorial_histogram_equalization.html
        """ 
        target_detected_ravel = self.target_detected.ravel()
        template_ravel = self.template.ravel()
        
        target_counts = np.bincount(target_detected_ravel, minlength = 256)
        target_quantiles = np.cumsum(target_counts).astype(np.float64)
        target_quantiles /= target_quantiles[-1]
        
        template_values =  np.arange(0, 256,1, dtype=np.uint8)
        template_counts = np.bincount(template_ravel, minlength = 256)
        template_quantiles = np.cumsum(template_counts).astype(np.float64)
        template_quantiles /= template_quantiles[-1]
        
        interp_template_values = np.interp(target_quantiles, template_quantiles, template_values)
        interp_template_values = interp_template_values.astype(self.target_image.dtype)
        
        self.target_image_corrected = interp_template_values[self.target_image_original]



    def measure(self, **kwargs):
        """Measure the pixel-to-mm-ratio of a reference card inside an image.
        
        Parameters
        ----------
        
        zoom_factor: int > 0 (default: 5)
            factor by which to magnify on scrollwheel use
        """
        ## kwargs 
        self.zoom_fac = kwargs.get("zoom_factor", 5)
#        if "ret" in kwargs:
#            ret_list = []
            
        ## for zoom
        self.image_height = self.template_image.shape[0]
        self.delta_height = int((self.image_height/self.zoom_fac)/2)
        self.image_width = self.template_image.shape[1]
        self.delta_width = int((self.image_width/self.zoom_fac)/2)
        self.image_diag = int((self.image_height + self.image_width)/2)
            
        ## initialize with empty lists / parameters
        self.done = False 
        self.flag_zoom = False
        self.current = (0, 0) 
        self.current_zoom = []
        self.points = []
        self.idx = 0
        self.idx_list = []
        self.distance_mm = ""
        
        ## feedback and setup        
        print("\nMeasure pixel-to-mm-ratio by clicking on two points, and type the distance in mm between them.")
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self._on_mouse_measure_scale)
        temp_canvas1 = copy.deepcopy(self.template_image)
        temp_canvas2 = copy.deepcopy(self.template_image)
        
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
                                    
            ## type distance onto image; accept only digits
            if k > 0 and k != 8 and k != 13 and k != 27:
                if chr(k).isdigit():
                    self.distance_mm = self.distance_mm + chr(k)
            elif k == 8:
                self.distance_mm = self.distance_mm[0:len(self.distance_mm)-1]
              
            ## measure distance in pixel
            if len(self.points) == 2:
                self.distance_px = int(math.sqrt(((self.points[0][0]-self.points[1][0])**2)+((self.points[0][1]-self.points[1][1])**2)))
            
            ## adjust text size and line width to canvas size            
            text_size = max(int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/500),1)
            text_thickness = max(int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/1000),1)
            linewidth = max(int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/500),1)
                                   
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
                cv2.putText(temp_canvas2, "Enter distance (mm): ", (text_mm_x, text_mm_y), cv2.FONT_HERSHEY_SIMPLEX, \
                            text_size, black, text_thickness, cv2.LINE_AA)
            elif len(self.distance_mm) > 0:
                cv2.putText(temp_canvas2, "Entered distance: " + self.distance_mm + " mm", (text_mm_x, text_mm_y), cv2.FONT_HERSHEY_SIMPLEX, \
                            text_size, green, int(text_thickness * 2), cv2.LINE_AA)
            
            ## text: scale measured
            if len(self.points) < 2:
                cv2.putText(temp_canvas2, "Mark scale (2 points): ", (text_px_x, text_px_y), cv2.FONT_HERSHEY_SIMPLEX, \
                            text_size, black, text_thickness, cv2.LINE_AA)
            elif len(self.points) == 2:
                cv2.putText(temp_canvas2, "Mark scale: " + str(self.distance_px) + " pixel", (text_px_x, text_px_y), cv2.FONT_HERSHEY_SIMPLEX, \
                            text_size, green, int(text_thickness * 2), cv2.LINE_AA)

            ## finish on enter, if conditions are met
            if k == 13:
                ## Warning if data needs to be entered
                if len(self.points) < 2:                  
                    cv2.putText(temp_canvas2, "No scale marked!", (int(temp_canvas2.shape[0]/5),int(temp_canvas2.shape[1]/3.5)), \
                                cv2.FONT_HERSHEY_SIMPLEX, text_size , red, int(text_thickness * 2), cv2.LINE_AA)
                if len(self.distance_mm) == 0:                  
                    cv2.putText(temp_canvas2, "No distance entered!", (int(temp_canvas2.shape[0]/5),int(temp_canvas2.shape[1]/2.5)), \
                                cv2.FONT_HERSHEY_SIMPLEX, text_size , red, int(text_thickness * 2), cv2.LINE_AA)               
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
        
#        if ret_list:


        return self.scale_ratio
        
    

    def create_template(self, **kwargs):
    
        """Make a template of a reference card inside an image for automatic detection and colour-correction.
        
        Parameters
        ----------
        
        zoom_factor: int > 0 (default: 5)
            factor by which to magnify on scrollwheel use
        mode: str (default: "rectangle")
            options: "rectangle" or "polygon"; draw the template with a bounding-polygon or a bounding-rectangle
        show: bool (default: True)
            show the resulting template with an overlay
        """
        
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
        
        ## for zoom
        self.image_height = self.template_image.shape[0]
        self.delta_height = int((self.image_height/self.zoom_fac)/2)
        self.image_width = self.template_image.shape[1]
        self.delta_width = int((self.image_width/self.zoom_fac)/2)
        self.image_diag = int((self.image_height + self.image_width)/2)
        
        ## feedback and setup        
        print("\nMark outline of scale-template in image, either by dragging a rectangle or a drawing polygon outline.")
        temp_canvas1 = copy.deepcopy(self.template_image)
        temp_canvas2 = copy.deepcopy(self.template_image)
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self._on_mouse_make_scale_template)

                
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
                    
            ## adjust text size and line width to canvas size            
            text_size = max(int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/1000),1)
            text_thickness = max(int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/1000),1)
            linewidth = max(int(((temp_canvas2.shape[0] + temp_canvas2.shape[1])/2)/500),1)

                    
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
                    if len(self.points) < 2:                  
                        cv2.putText(temp_canvas2, "At least two points needed to draw rectangle!", (int(temp_canvas2.shape[0]/2.5),int(temp_canvas2.shape[1]/3.5)), cv2.FONT_HERSHEY_SIMPLEX, text_size, red, int(text_thickness * 2), cv2.LINE_AA)
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
            
        print("\n")
        print("------------------------------------------------")
        print("Finished - scale template completed")
        print("------------------------------------------------")
        print("\n")
            
        # create bounding box and cut template
        x,y,w,h = cv2.boundingRect(np.array(self.points))
        self.points_bounding_rect = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
        self.template = self.template_image[y:y+h,x:x+w]

        # create colour mask to show template 
        self.colour_mask = np.zeros(self.template_image.shape, np.uint8)
        self.colour_mask[:,:,1] = 255 # all area green
        cv2.fillPoly(self.colour_mask, np.array([self.points_bounding_rect]), red) # red = excluded area
        self.overlay = cv2.addWeighted(copy.deepcopy(self.template_image), .7, self.colour_mask, 0.3, 0) # combine
        if self.mode=="polygon":
            points_drawing = self.points 
            points_drawing.append(self.points[0])
            cv2.polylines(self.overlay, np.array([points_drawing]), False, green, linewidth)
        if self.mode=="rectangle":
            cv2.rectangle(self.overlay, self.points[0], self.points[1], green, linewidth)

        ## show template
        if self.show == True:
            show_img(self.overlay)
        
        ## create exact mask for scale detection
        if self.mode=="polygon":
            self.template_mask = np.zeros(self.template_image.shape[0:2], np.uint8)
            cv2.fillPoly(self.template_mask, [np.array(self.points)], white) 
            self.template_mask = self.template_mask[y:y+h,x:x+w]
        elif self.mode=="rectangle":
            self.template_mask = np.zeros(self.template_image.shape[0:2], np.uint8)
            self.template_mask = self.template_mask[y:y+h,x:x+w]
            self.template_mask.fill(255)
            
        ## create reference circle to size adjust detected scale-cards
        (x,y),radius = cv2.minEnclosingCircle(np.array(self.points))
        self.reference_diameter = (radius * 2)
       
        ## create mask object for template image (same proporties as mask_maker objects)
        zeros = np.zeros(self.template_image.shape[0:2], np.uint8)
        if self.mode=="polygon":
            self.template_image_mask = cv2.fillPoly(zeros, [np.array(self.points)], white)
        elif self.mode=="rectangle":
            self.template_image_mask = cv2.fillPoly(zeros, [np.array(self.points_bounding_rect)], white)
        self.template_image_mask = np.array(self.template_image_mask, dtype=bool)

        ## return scale template and mask object (boolean mask of full image, label, include-flag)
        return self.template, (self.template_image_mask, "scale-template", False)


    
    def detect(self, target_image, **kwargs):
        """Find scale from a defined template inside an image and update pixel ratio. Image registration is run by the "AKAZE" algorithm 
        (http://www.bmva.org/bmvc/2013/Papers/paper0013/abstract0013.pdf). Future implementations will include more algorithms to select from.
        Prior to running detect_scale, measure_scale and make_scale_template have to be run once to pass on reference scale size and template of 
        scale reference card (gets passed on internally by calling detect_scale from the same instance of scale_maker.
        
        Parameters
        -----------
        target_image: str or array
            absolute or relative path to OR numpy array of targeted image that contains the scale reference card
        resize: num (optional, default: 1 or 0.5 for images with diameter > 5000px)
            resize image to speed up detection process (WARNING: too low values may result in poor detection results or even crashes)
        show: bool (optional, default: False)
            show result of scale detection procedure on current image  
        """      
        ## kwargs
        min_matches = kwargs.get('min_matches', 10)
        flag_show = kwargs.get('show', False)
        flag_equalize = kwargs.get('equalize', False)

        ## initialize
        if isinstance(target_image, str):
            self.target_image = cv2.imread(target_image)
        else:
            self.target_image = target_image
            
        ## if "self.measure" is skipped
        if not hasattr(self, "scale_ratio"):
            self.scale_ratio = 1
            
        self.target_image_original = copy.deepcopy(self.target_image)
            
        if not len(self.target_image.shape)==3:
            self.target_image = cv2.cvtColor(self.target_image, cv2.COLOR_GRAY2BGR)
            
        ## if image diameter bigger than 2000 px, then automatically resize
        if (self.target_image.shape[0] + self.target_image.shape[1])/2 > 2000:
            self.resize_factor = kwargs.get('resize', 0.5)
        else:
            self.resize_factor = kwargs.get('resize', 1)
        self.target_image = cv2.resize(self.target_image, (0,0), fx=1*self.resize_factor, fy=1*self.resize_factor) 

        # =============================================================================
        # AKAZE detector
        # =============================================================================     
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(self.template, self.template_mask)
        kp2, des2 = akaze.detectAndCompute(self.target_image, None)       
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
            ret, contours, hierarchy = cv2.findContours(self.template_mask,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_TC89_L1)
            box = contours[0].astype(np.float32)

            rect  = cv2.perspectiveTransform(box,M).astype(np.int32)
            self.target_image = cv2.polylines(self.target_image,[rect],True,red,5, cv2.LINE_AA)

            # update current scale using reference
            rect = rect/self.resize_factor
            rect = np.array(rect, dtype=np.int32)
            (x,y),radius = cv2.minEnclosingCircle(rect)
            self.detected_diameter = (radius * 2)
            self.diff_ratio = (self.detected_diameter / self.reference_diameter)
            self.current = round(self.scale_ratio * self.diff_ratio,1)
            
            ## resize target image back to original size
            self.target_image = cv2.resize(self.target_image, (0,0), fx=1/self.resize_factor, fy=1/self.resize_factor) 
            
            # create mask of detected scale reference card
            zeros = np.zeros(self.target_image.shape[0:2], np.uint8)
            mask_bin = cv2.fillPoly(zeros, [np.array(rect)], white)       
            self.detected_mask = np.array(mask_bin, dtype=bool)
            
            # cut out target reference card
            (rx,ry,w,h) = cv2.boundingRect(rect)
            self.target_detected = self.target_image_original[ry:ry+h,rx:rx+w]
            
            print("\n")
            print("---------------------------------------------------")
            print("Reference card found with %d keypoint matches:" % self.nkp)
            print("current image has %s pixel per mm." % (self.scale_ratio))
            print("= %s %% of template image." % round(self.diff_ratio * 100,3))
            print("---------------------------------------------------")
            print("\n")
            
            ## do histogram equalization
            if flag_equalize:
                self._equalize()

            ## show results
            if flag_show:
                show_img([self.template_image, self.target_image, self.target_image_corrected])
                
            return self.target_image_corrected, (self.detected_mask, "scale-detected", False), self.current       

        
        else:
            print("\n")
            print("---------------------------------------------------")
            print("Reference card not found - only %d/%d keypoint matches" % (self.nkp, min_matches))
            print("---------------------------------------------------")
            print("\n")
            
            return "no current scale", "no scale mask"



def create_mask(image, **kwargs):
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
        
    ## load image
    if isinstance(image, str):
        image = cv2.imread(image)  
    if len(image.shape)==2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    label = kwargs.get("label","area 1")
    max_dim = kwargs.get("max_dim",1980)
    include = kwargs.get("include",True)
    
    flag_show = kwargs.get("show",False)
    flag_tool = kwargs.get("tool","rectangle")

    print("\nMark the outline of your arena, i.e. what you want to include in the image analysis by left clicking, finish with enter.")

    iv_object = _image_viewer(image, mode="interactive", max_dim = max_dim, tool=flag_tool)
    
    zeros = np.zeros(image.shape[0:2], np.uint8)
    
    if flag_tool == "rectangle" or flag_tool == "box":
        for rect in iv_object.rect_list:
            pts = np.array(((rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])), dtype=np.int32)
            mask_bin = cv2.fillPoly(zeros, [pts], white)
    elif flag_tool == "polygon" or flag_tool == "free":
        for poly in iv_object.poly_list:
            pts = np.array(poly, dtype=np.int32)
            mask_bin = cv2.fillPoly(zeros, [pts], white)

    if include == False:
        mask_bool = np.invert(mask_bin)
    mask_bool = np.array(mask_bin, dtype=bool)

    overlay = np.zeros(image.shape, np.uint8) # make overlay
    overlay[:,:,2] = 200 # start with all-red overlay
    overlay[mask_bool,1] = 200   
    overlay[mask_bool,2] = 0   
    overlay = cv2.addWeighted(image, .7, overlay, 0.5, 0)
    
    if flag_show:
        show_img(overlay)
    if flag_tool == "rectangle" or flag_tool == "box":
        mask_list = iv_object.rect_list
    elif flag_tool == "polygon" or flag_tool == "free":
        mask_list = iv_object.poly_list

    class mask_object:
        def __init__(self, label, overlay, zeros, mask_bool, mask_list):
            self.label = label
            self.overlay = overlay
            self.mask_bin = zeros
            self.mask_bool = mask_bool
            self.mask_list = mask_list
    
    return mask_object(label, overlay, zeros, mask_bool, mask_list)  # Create an empty record


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
        self.image_orig = copy.deepcopy(self.image)
        
            
    def find_objects(self, **kwargs):    
        """Method in object finder class: find objects in colour or grayscale images using thresholding
        
        Parameters
        ----------
        filename: str (default: "My file")
            filename of image to find objects in
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
        ## kwargs
        self.resize_factor = kwargs.get("resize", 1)
        show = kwargs.get('show', True)
        mask_objects = kwargs.get('mask', [])

        self.filename = "My file"

        self.image = cv2.resize(self.image_orig, (0,0), fx=1*self.resize_factor, fy=1*self.resize_factor) 
        
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
            if self.resize_factor > 0.25:
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

    
        # =============================================================================
        # masking
        # =============================================================================
        
        if len(mask_objects)>0:
                
            mask_dummy1 = np.zeros(self.morph.shape)
            
            for obj in mask_objects:
                mask, label, include = obj
                mask_bin = np.array(mask, dtype=np.uint8)
                mask_bin = cv2.resize(mask_bin, (0,0), fx=1*self.resize_factor, fy=1*self.resize_factor)
                mask = np.array(mask_bin, dtype=bool)
                if include == True:
                    mask_dummy2 = np.zeros(self.morph.shape)
                    mask_dummy2[mask] = 1
                    mask_dummy1 = np.add(mask_dummy1, mask_dummy2)
                if include == False:
                    mask_dummy2 = np.zeros(self.morph.shape)
                    mask_dummy2.fill(255)
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
                            cv2.drawContours(self.image_processed, [cnt], 0, red, int(10 * self.resize_factor))
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
                    cv2.drawContours(self.image_processed, [cnt], 0, red, int(10 * self.resize_factor))
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
            self.df.insert(3, "resize_factor", self.resize_factor)
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

