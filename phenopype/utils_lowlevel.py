#%% modules
import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd

from datetime import datetime
from stat import S_IWRITE
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ordereddict
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from phenopype import presets
from phenopype.settings import colours

#%% classes

class _image_viewer():
    def __init__(self, image, **kwargs):
        """
        

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ## kwargs
        self.flag_tool = kwargs.get("tool", None)
        self.flag_zoom_mode = kwargs.get("zoom", "continuous")
        self.zoom_mag = kwargs.get("mag", 0.7)
        self.zoom_n_steps = kwargs.get("steps", 20)
        self.window_name = kwargs.get("window_name", "phenopype")
        window_aspect = kwargs.get("window_aspect", cv2.WINDOW_AUTOSIZE)
        window_control = kwargs.get("window_control", "internal")
        window_max_dimension = kwargs.get("max_dim", 1000)

        ## resize image canvas
        image_width, image_height = image.shape[1], image.shape[0]
        if image_height > window_max_dimension or image_width > window_max_dimension:
            if image_width > image_height:
                canvas = cv2.resize(image, (window_max_dimension, int((window_max_dimension/image_width) * image_height)), cv2.INTER_AREA)
            else:
                canvas = cv2.resize(image, (int((window_max_dimension/image_height) * image_width), window_max_dimension), cv2.INTER_AREA)
        else:
            canvas = copy.deepcopy(image)

        ## initialize
        self.image = copy.deepcopy(image)
        self.image_copy = copy.deepcopy(image)
        self.canvas = copy.deepcopy(canvas)
        self.canvas_copy = copy.deepcopy(canvas)
        self.canvas_width, self.canvas_height = self.canvas.shape[1], self.canvas.shape[0]
        self.image_width, self.image_height = self.image.shape[1], self.image.shape[0]
        self.canvas_fx, self.canvas_fy =  self.image_width/self.canvas_width, self.image_height/self.canvas_height
        self.global_fx, self.global_fy = self.canvas_fx , self.canvas_fy 

        ## zoom config
        self.zoom_x1, self.zoom_y1, self.zoom_x2, self.zoom_y2 = 0, 0, self.image_width, self.image_height
        self.flag_zoom, self.zoom_idx = -1, 1
        self.zoom_step_x, self.zoom_step_y = int(image_width/self.zoom_n_steps), int(image_height/self.zoom_n_steps)
        if self.flag_zoom_mode == "fixed":
            mag = int(self.zoom_mag * self.zoom_n_steps)
            self.zoom_step_x, self.zoom_step_y = mag * self.zoom_step_x , mag * self.zoom_step_y

        ## interactive window
        if self.flag_tool:
            self.points, self.point_list = [], []
            self.line_width = kwargs.get("line_width", _auto_line_width(image))
            if self.flag_tool==self.flag_tool == "rectangle" or self.flag_tool == "box":
                self.rect_list, self.rect_start = [], None
            elif self.flag_tool == "landmarks" or self.flag_tool == "landmark":
                self.point_size = kwargs.get("point_size", _auto_point_size(image))
                self.point_col = colours[kwargs.get("point_col", "red")]
                self.text_size = kwargs.get("label_size", _auto_text_size(image))
                self.text_width = kwargs.get("label_size", _auto_text_width(image))
                self.text_col = colours[kwargs.get("label_col", "black")]
            
        ## update from a previous call
        if kwargs.get("prev_attributes"):
            prev_attr = kwargs.get("prev_attributes")
            prev_attr = {i:prev_attr[i] for i in prev_attr if i not in ["canvas_copy", "canvas", "image_copy","image"]}
            self.__dict__.update(prev_attr)
            if hasattr(self, "rect_list"):
                for (rx1, ry1, rx2, ry2) in self.rect_list:
                            cv2.rectangle(self.image_copy, (rx1,ry1), (rx2,ry2), colours["green"], self.line_width)
            if hasattr(self, "poly_list"):
                for poly in self.point_list:
                    cv2.polylines(self.image_copy, np.array([poly]), False, colours["green"], self.line_width)
            self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2,]
            self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
            self.canvas_copy = copy.deepcopy(self.canvas)

        ## show canvas
        self.done = False
        cv2.namedWindow(self.window_name, window_aspect)
        cv2.startWindowThread() ## needed for Mac OS ??
        cv2.setMouseCallback(self.window_name, self._on_mouse_plain)
        cv2.resizeWindow(self.window_name, self.canvas_width, self.canvas_height)
        cv2.imshow(self.window_name, self.canvas)
        
        ## window control
        if window_control=="internal":
            if cv2.waitKey() == 13:
                self.done = True
                cv2.destroyAllWindows()
                if self.flag_tool == "polygon" or self.flag_tool == "free":
                    if len(self.points)>2:
                        self.points.append(self.points[0])
                        self.point_list.append(self.points)
                elif self.flag_tool == "rectangle" or self.flag_tool == "box":
                    if len(self.rect_list)>0:
                        for rect in self.rect_list:
                            xmin, ymin, xmax, ymax = rect
                            self.point_list.append([(xmin, ymin), (xmax,ymin), (xmax, ymax), (xmin, ymax)])
                elif self.flag_tool == "landmarks" or self.flag_tool == "landmark":
                    self.point_list.append(self.points)
            elif cv2.waitKey() == 27:
                cv2.destroyAllWindows()
                sys.exit("User intterupt - closing phenopype window")

    def _on_mouse_plain(self, event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEWHEEL and flags > 0:
            if self.zoom_idx < self.zoom_n_steps:
                self.flag_zoom = 1
                self.zoom_idx += 1
                if self.flag_zoom_mode == "continuous" or (self.flag_zoom_mode == "fixed" and self.zoom_idx == 2):
                    self._zoom_fun(x,y)
                self.x, self.y = x, y    
                cv2.imshow(self.window_name, self.canvas)
        if event == cv2.EVENT_MOUSEWHEEL and flags < 0:
            if self.zoom_idx > 1:
                self.flag_zoom = -1
                self.zoom_idx -= 1
                if self.flag_zoom_mode == "continuous" or (self.flag_zoom_mode == "fixed" and self.zoom_idx == 1):
                    self._zoom_fun(x,y)
                self.x, self.y = x, y    
                cv2.imshow(self.window_name, self.canvas)
        if self.flag_tool:
            if self.flag_tool == "landmark" or self.flag_tool == "landmarks":
                self._on_mouse_landmark(event, x, y)
            if self.flag_tool == "rectangle" or self.flag_tool == "box":
                self._on_mouse_rectangle(event, x, y, flags)
            elif self.flag_tool == "polygon" or self.flag_tool == "free":
                self._on_mouse_polygon(event, x, y, flags)

    def _on_mouse_landmark(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN: ## and (flags & cv2.EVENT_FLAG_CTRLKEY)
            self.coords_original = int(self.zoom_x1+(x * self.global_fx)), int(self.zoom_y1+(y * self.global_fy))
            self.points.append(self.coords_original)
            cv2.circle(self.image_copy, self.coords_original, self.point_size, self.point_col, -1)
            cv2.putText(self.image_copy, str(len(self.points)), self.coords_original, 
                        cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_col, self.text_width, cv2.LINE_AA)
            self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
            self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
            self.canvas_copy = copy.deepcopy(self.canvas)
            cv2.imshow(self.window_name, self.canvas)
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points)>0:
                self.points = self.points[:-1]
                self.image_copy = copy.deepcopy(self.image)
                for point, idx in zip(self.points, range(len(self.points))):
                    cv2.circle(self.image_copy, point, self.point_size, self.point_col, -1)
                    cv2.putText(self.image_copy, str(idx+1), point, 
                        cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_col, self.text_width, cv2.LINE_AA)
                self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                self.canvas_copy = copy.deepcopy(self.canvas)
                cv2.imshow(self.window_name, self.canvas)
                
    def _on_mouse_polygon(self, event, x, y, flags):
        if event == cv2.EVENT_MOUSEMOVE:
            self.coords_original = int(self.zoom_x1+(x * self.global_fx)), int(self.zoom_y1+(y * self.global_fy))
            if len(self.points) > 0:
                self.coords_prev = int((self.points[-1][0]-self.zoom_x1)/self.global_fx), int((self.points[-1][1]-self.zoom_y1)//self.global_fy)
                self.canvas = copy.deepcopy(self.canvas_copy)
                cv2.line(self.canvas, self.coords_prev, (x,y), colours["blue"], self.line_width)
            cv2.imshow(self.window_name, self.canvas)
        if event == cv2.EVENT_LBUTTONDOWN: ## and (flags & cv2.EVENT_FLAG_CTRLKEY)
            self.coords_original = int(self.zoom_x1+(x * self.global_fx)), int(self.zoom_y1+(y * self.global_fy))
            self.points.append(self.coords_original)
            cv2.polylines(self.image_copy, np.array([self.points]), False, colours["green"], self.line_width)
            if len(self.point_list)>0:
                for poly in self.point_list:
                    cv2.polylines(self.image_copy, np.array([poly]), False, colours["green"], self.line_width)
            self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
            self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
            self.canvas_copy = copy.deepcopy(self.canvas)
            cv2.imshow(self.window_name, self.canvas)
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points)>0:
                self.points = self.points[:-1]
                self.image_copy = copy.deepcopy(self.image)
                cv2.polylines(self.image_copy, np.array([self.points]), False, colours["green"], self.line_width)
                for poly in self.point_list:
                    cv2.polylines(self.image_copy, np.array([poly]), False, colours["green"], self.line_width)
                self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                self.canvas_copy = copy.deepcopy(self.canvas)
                cv2.imshow(self.window_name, self.canvas)
            elif len(self.points) == 0 and len(self.point_list)>0:
                self.point_list = self.point_list[:-1]
                self.image_copy = copy.deepcopy(self.image)
                for poly in self.point_list:
                    cv2.polylines(self.image_copy, np.array([poly]), False, colours["green"], self.line_width)
                self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                self.canvas_copy = copy.deepcopy(self.canvas)
                cv2.imshow(self.window_name, self.canvas)
        if flags == cv2.EVENT_FLAG_CTRLKEY and len(self.points)>2:
            self.points.append(self.points[0])
            self.point_list.append(self.points)
            self.points = []
            self.image_copy = copy.deepcopy(self.image)
            if len(self.point_list)>0:
                for poly in self.point_list:
                    cv2.polylines(self.image_copy, np.array([poly]), False, colours["green"], self.line_width)
            self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
            self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
            self.canvas_copy = copy.deepcopy(self.canvas)

    def _on_mouse_rectangle(self, event, x, y, flags):
            if event == cv2.EVENT_LBUTTONDOWN: ## and (flags & cv2.EVENT_FLAG_CTRLKEY)
                self.rect_start = x, y
                self.canvas_copy = copy.deepcopy(self.canvas)
            if event == cv2.EVENT_LBUTTONUP: ## and (flags & cv2.EVENT_FLAG_CTRLKEY)
                self.rect_start = None
                self.rect_list.append([
                        int(self.zoom_x1 + (self.global_fx * self.rect_minpos[0])), 
                        int(self.zoom_y1 + (self.global_fy * self.rect_minpos[1])),
                        int(self.zoom_x1 + (self.global_fx * self.rect_maxpos[0])), 
                        int(self.zoom_y1 + (self.global_fy * self.rect_maxpos[1]))])
                for (rx1, ry1, rx2, ry2) in self.rect_list:
                    cv2.rectangle(self.image_copy, (rx1,ry1), (rx2,ry2), colours["green"], self.line_width)
                self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                self.canvas_copy = copy.deepcopy(self.canvas)
                cv2.imshow(self.window_name, self.canvas)
            if event == cv2.EVENT_RBUTTONDOWN:
                if len(self.rect_list)>0:
                    self.rect_list = self.rect_list[:-1]
                    self.image_copy = copy.deepcopy(self.image)
                    for (rx1, ry1, rx2, ry2) in self.rect_list:
                        cv2.rectangle(self.image_copy, (rx1,ry1), (rx2,ry2), colours["green"], self.line_width)
                    self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                    self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                    self.canvas_copy = copy.deepcopy(self.canvas)
                    cv2.imshow(self.window_name, self.canvas)
            elif self.rect_start:
                if flags & cv2.EVENT_FLAG_LBUTTON: ##  and (flags & cv2.EVENT_FLAG_CTRLKEY)
                    self.canvas = copy.deepcopy(self.canvas_copy)
                    self.rect_minpos = min(self.rect_start[0], x), min(self.rect_start[1], y)
                    self.rect_maxpos = max(self.rect_start[0], x), max(self.rect_start[1], y)
                    cv2.rectangle(self.canvas, self.rect_minpos, self.rect_maxpos, 
                                  colours["red"], max(2,_auto_line_width(self.canvas)))
                    cv2.imshow(self.window_name, self.canvas)

    def _zoom_fun(self,x,y):
        """Helper function for image_viewer. Takes current xy coordinates and zooms in within a rectangle around mouse coordinates. 
        Transforms current coordinates back to original coordinate space
        """
        if y <= 0:
            y = 1
        if x <= 0:
            x = 1
            
        x_prop, y_prop = x/self.canvas_width, y/self.canvas_height
        left_padding, right_padding = int(round(x_prop * self.zoom_step_x)), int(round((1-x_prop) * self.zoom_step_x))
        top_padding, bottom_padding = int(round(y_prop * self.zoom_step_y)), int(round((1-y_prop) * self.zoom_step_y))

        if self.flag_zoom > 0:
            x1, x2 = self.zoom_x1 + left_padding, self.zoom_x2 - right_padding 
            y1, y2 = self.zoom_y1 + top_padding, self.zoom_y2 - bottom_padding
        if self.flag_zoom < 0:
            x1, x2 = self.zoom_x1 - left_padding, self.zoom_x2 + right_padding, 
            y1, y2 = self.zoom_y1 - top_padding, self.zoom_y2 + bottom_padding
            if x1 < 0:
                x2 = x2 + abs(x1)
                x1 = 0
            if x2 > self.image_width:
                x1 = x1 - (x2 - self.image_width)
                x2 = self.image_width
            if y1 < 0:
                y2 = y2 + abs(y1)
                y1 = 0
            if y2 > self.image_height:
                y1 = y1 - (y2 - self.image_height)
                y2 = self.image_height

        self.zoom_x1, self.zoom_x2, self.zoom_y1, self.zoom_y2 = x1, x2, y1, y2
        
        self.global_fx = self.canvas_fx * ((self.zoom_x2-self.zoom_x1)/self.image_width) 
        self.global_fy = self.canvas_fy * ((self.zoom_y2-self.zoom_y1)/self.image_height)
        
        self.canvas = self.image_copy[y1:y2,x1:x2]
        self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
        self.canvas_copy = copy.deepcopy(self.canvas)



class _yaml_file_monitor:
    def __init__(self, filepath, **kwargs):

        ## kwargs       
        self.flag_print = kwargs.get("print_settings", False)

        ## file, location and event action        
        self.dirpath = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)
        self.filepath = filepath
        self.event_handler = PatternMatchingEventHandler(patterns=["*/" + self.filename])
        self.event_handler.on_any_event = self.on_update
        
        ## intitialize
        self.content = _load_yaml(self.filepath)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.dirpath, recursive=False)
        self.observer.start()

    def on_update(self, event):
        self.content = _load_yaml(self.filepath)
        if self.flag_print == True:
            print(_show_yaml(self.content), end="")
        self.flag_update = True
        cv2.destroyAllWindows()

    def stop(self):
        self.observer.stop()
        self.observer.join()

#%% functions

def _auto_line_width(image, **kwargs):
    factor = kwargs.get("factor", 0.001)
    image_height,image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) /2
    line_tickness = int(factor * image_diagonal)

    return line_tickness



def _auto_point_size(image, **kwargs):
    factor = kwargs.get("factor", 0.002)
    image_height,image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) /2
    point_size = int(factor * image_diagonal)

    return point_size



def _auto_text_width(image, **kwargs):
    factor = kwargs.get("factor", 0.0005)
    image_height,image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) /2
    text_tickness = int(factor * image_diagonal)

    return text_tickness



def _auto_text_size(image, **kwargs):
    factor = kwargs.get("factor", 0.00025)
    image_height,image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) /2
    text_size = int(factor * image_diagonal)

    return text_size



def _create_mask_bin(image, coords):
    mask_bin = np.zeros(image.shape[0:2], np.uint8)
    for sub_coords in coords:
        cv2.fillPoly(mask_bin, [np.array(sub_coords, dtype=np.int32)], colours["white"])
    return mask_bin


def _create_mask_bool(image, coords):
    mask_bin = np.zeros(image.shape[0:2], np.uint8)
    for sub_coords in coords:
        cv2.fillPoly(mask_bin, [np.array(sub_coords, dtype=np.int32)], colours["white"])
    return np.array(mask_bin, dtype=bool)


def _del_rw(action, name, exc):
    os.chmod(name, S_IWRITE)
    os.remove(name)



def _file_walker(directory, **kwargs):
    """
    
    Parameters
    ----------
    directory : TYPE
        DESCRIPTION.
    search_mode (optional): str (default: "dir")
        "dir" searches current directory for valid files; "recursive" walks through all subdirectories
    filetypes (optional): list of str
        single or multiple string patterns to target files with certain endings
    include (optional): list of str
        single or multiple string patterns to target certain files to include
    exclude (optional): list of str
        single or multiple string patterns to target certain files to exclude - can overrule "include"
    unique_mode (optional): str (default: "filepath")
        how should unique files be identified: "filepath" or "filename". "filepath" is useful, for example, 
        if identically named files exist in different subfolders (folder structure will be collapsed and goes into the filename),
        whereas filename will ignore all those files after their first occurrence.

    Returns
    -------
    None.

    """
    ## kwargs
    search_mode = kwargs.get("search_mode","dir")
    unique_mode = kwargs.get("unique_mode", "filepath")
    filetypes = kwargs.get("filetypes", [])
    if not filetypes.__class__.__name__ == "list":
        filetypes = [filetypes]
    include = kwargs.get("include", [])
    if not include.__class__.__name__ == "list":
        include = [include]
    exclude = kwargs.get("exclude", [])
    if not exclude.__class__.__name__ == "list":
        exclude = [exclude]


    ## find files 
    filepaths1, filepaths2, filepaths3, filepaths4 = [],[],[],[]
    
    if search_mode == "recursive":
        for root, dirs, files in os.walk(directory):
            for file in os.listdir(root):
                filepath = os.path.join(root,file)
                if os.path.isfile(filepath):
                    filepaths1.append(filepath)
    elif search_mode == "dir":
        for file in os.listdir(directory):
            filepath = os.path.join(directory,file)
            if os.path.isfile(filepath):   
                filepaths1.append(filepath)

    ## file endings
    if len(filetypes)>0:
        for filepath in filepaths1:
            if filepath.endswith(tuple(filetypes)):
                filepaths2.append(filepath)
    elif len(filetypes)==0:
        filepaths2 = filepaths1

    ## include
    if len(include)>0:
        for filepath in filepaths2:   
            if any(inc in os.path.basename(filepath) for inc in include):
                filepaths3.append(filepath)
    else:
        filepaths3 = filepaths2

    ## exclude
    if len(exclude)>0:
        for filepath in filepaths3:   
            if not any(exc in os.path.basename(filepath) for exc in exclude):
                filepaths4.append(filepath)
    else:
        filepaths4 = filepaths3

    ## check if files found
    filepaths = filepaths4
    if len(filepaths) == 0:
        sys.exit("No files found under the given location that match given.")

    ## allow unique filenames filepath or by filename only
    filenames, unique_filename, unique, duplicate = [],[],[],[]
    for filepath in filepaths:
        filenames.append(os.path.basename(filepath))
        
    if unique_mode=="filepaths" or unique_mode=="filepath":
        for filename, filepath in zip(filenames, filepaths):
            if not filepath in unique:
                unique.append(filepath)
            else:
                duplicate.append(filepath)
    elif unique_mode=="filenames" or unique_mode=="filename":
        for filename, filepath in zip(filenames, filepaths):
            if not filename in unique_filename:
                unique_filename.append(filename)
                unique.append(filepath)
            else:
                duplicate.append(filepath)

    return unique, duplicate



def _load_masks(obj_input, mask_list):
    if obj_input.__class__.__name__ == "ndarray":
        if mask_list.__class__.__name__ == "dict" or mask_list.__class__.__name__ == "CommentedMap":
            masks = []
            for mask in  list(mask_list.items()):
                masks.append(mask[1])
        elif mask_list.__class__.__name__ == "str":
            masks = []
        elif mask_list.__class__.__name__ == "NoneType":
            masks = []
    elif obj_input.__class__.__name__ == "container":
        if mask_list.__class__.__name__ == "dict" or mask_list.__class__.__name__ == "CommentedMap":
            masks = []
            for mask in  list(mask_list.items()):
                masks.append(mask[1])
        elif mask_list.__class__.__name__ == "list" or mask_list.__class__.__name__ == "CommentedSeq":
            if all(isinstance(n, dict) for n in mask_list):
                masks = mask_list            
            elif all(isinstance(n, str) for n in mask_list):
                masks = []
                for mask in mask_list:
                    if mask in obj_input.masks:
                        masks.append(obj_input.masks[mask])
        elif mask_list.__class__.__name__ == "str":
            if obj_input.masks:
                if mask_list in obj_input.masks:
                    masks = [obj_input.masks[mask_list]]
                else:
                    masks = []
            else:
                masks = []
        elif mask_list.__class__.__name__ == "NoneType" and len(obj_input.masks) > 0: ## too confusing?
            masks, mask_list = list(obj_input.masks.values()), list(obj_input.masks.keys())
        elif mask_list.__class__.__name__ == "NoneType":
            masks = []
    return masks, mask_list



def _load_yaml(string):
    yaml = YAML()
    if os.path.isfile(string):
        with open(string, 'r') as file:
            file = yaml.load(file)
        return file
    else:
        warnings.warn("Not a valid path - couldn't load yaml.")
        # file = yaml.load(string)




def _load_pype_config(container, **kwargs):
    
    ## kwargs
    config = kwargs.get("config", "pype_generic.yaml")

    dirpath = container.dirpath    
    if os.path.isfile(config):
        return _load_yaml(config), config
    
    pype_config_locations = [os.path.join(dirpath, "pype_" + config + ".yaml"),
                             os.path.join(dirpath, config + ".yaml"),
                             os.path.join(dirpath, config)]
    for location in pype_config_locations:
        if os.path.isfile(location):
            return _load_yaml(location), location

    warnings.warn("No pype configuration found under given name - defaulting to preset1")
    pype_preset, location = _generic_pype_config(container)
    return pype_preset, location



def _generic_pype_config(container, **kwargs):

    ## kwargs
    preset = kwargs.get("preset", "preset1")
    
    dirpath = container.dirpath
    location = os.path.join(dirpath, "pype_generic.yaml")
    config = _load_yaml(eval("presets." + preset))
    pype_preset = {"image": container.image_data,
                   "pype":
                       {"name": "pype_generic",
                        "preset": "preset1",
                        "date_created": datetime.today().strftime('%Y%m%d_%H%M%S'),
                        "date_last_used": None}}
    pype_preset.update(config)
    print("Generic pype config generated from " + preset + ".")
    return pype_preset, location



def _show_yaml(ordereddict):
    yaml = YAML()
    yaml.dump(ordereddict, sys.stdout)



def _save_yaml(ordereddict, filepath):
    with open(filepath, 'w') as config_file:
        yaml = YAML()
        yaml.dump(ordereddict, config_file)
        
# def get_median_grayscale(image, **kwargs):
#     if (image.shape[0] + image.shape[1])/2 > 2000:
#         factor = kwargs.get('resize', 0.5)
#         image = cv2.resize(image, (0,0), fx=1*factor, fy=1*factor) 
        
#     vector = np.ravel(image)
#     vector_mc = Counter(vector).most_common(9)
#     g = [item[0] for item in vector_mc]
#     return int(np.median(g))

# def avgit(x):
#     return x.sum(axis=0)/np.shape(x)[0]

# def decode_fourcc(cc):
#     return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])