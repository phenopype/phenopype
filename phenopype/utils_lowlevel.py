#%% modules
import cv2
import copy
import datetime
import os
import stat
import sys
import numpy as np

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ordereddict

from phenopype.settings import colours, pype_presets


#%% methods

def _auto_line_thickness(image, **kwargs):
    factor = kwargs.get("factor", 0.001)
    image_height,image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) /2
    line_tickness = int(factor * image_diagonal)

    return line_tickness

def _auto_text_thickness(image, **kwargs):
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

def _load_image(obj_input, **kwargs):
    
    
    ##kwargs 
    flag_load = kwargs.get("load","mod")
    
    ## load image from path
    if obj_input.__class__.__name__ == "str":
        flag_input = "str"
        if os.path.isfile(obj_input):
            image = cv2.imread(obj_input)
        else:
            sys.exit("invalid image path")

    elif obj_input.__class__.__name__ == "ndarray":
        flag_input = "ndarray"
        image = obj_input
    

        
    elif obj_input.__class__.__name__ == "pype_container":
        flag_input = "pype_container"
        if flag_load == "mod":
            image = copy.deepcopy(obj_input.image_mod)
        elif flag_load == "canvas":
            if obj_input.canvas.__class__.__name__ == "ndarray":
                image = copy.deepcopy(obj_input.canvas)
            else:
                image = copy.deepcopy(obj_input.image)
        elif flag_load == "raw":
            image = copy.deepcopy(obj_input.image)
    return image, flag_input



def _make_pype_template(name, preset, **kwargs):
    
    ## kwargs
    pype_name = name
    preset_name = preset
    
    image_data = kwargs.get("attributes_file", None)
        
    ## load preset
    # pype_config = ordereddict([('pype_header', 
    #                     [ordereddict([('name', pype_name)]), 
    #                      ordereddict([('preset', preset_name)]), 
    #                      ordereddict([('created_on', datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))]), 
    #                      ordereddict([('last_used_on', datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))]), 
    #                      ordereddict([('modified_on', datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))])])])
    
    
    pype_config = {"pype_info":
                   {"name": pype_name,
                    "preset": preset_name,
                    "date_created": datetime.datetime.today().strftime('%Y%m%d_%H%M%S'),
                    "date_modified": datetime.datetime.today().strftime('%Y%m%d_%H%M%S')}
                   }
    yaml = YAML()
    preset = yaml.load(eval("pype_presets." + preset_name))
    pype_config.update(preset)

    return pype_config


def _del_rw(action, name, exc):
    os.chmod(name, stat.S_IWRITE)
    os.remove(name)



class _image_viewer():
    def __init__(self, image, **kwargs):

        ## kwargs
        self.flag_zoom_mode = kwargs.get("zoom", "continuous")
        self.flag_mode = kwargs.get("mode", None)
        self.flag_tool = kwargs.get("tool", "rectangle")
        self.ret_zoom_attributes = kwargs.get("ret_zoom_attributes", False)
        self.window_name = kwargs.get("window_name", "phenopype")
        self.window_aspect = kwargs.get("window_aspect", cv2.WINDOW_AUTOSIZE)
        self.window_control = kwargs.get("window_control", "internal")
        
        ## load image
        if isinstance(image, str):
            image = cv2.imread(image)  

        ## resize image canvas
        image_width, image_height = image.shape[1], image.shape[0]
        max_dim = kwargs.get("max_dim", 1000)
        if image_height > max_dim or image_width > max_dim:
            if image_width > image_height:
                canvas = cv2.resize(image, (max_dim, int((max_dim/image_width) * image_height)), cv2.INTER_AREA)
            else:
                canvas = cv2.resize(image, (int((max_dim/image_height) * image_width), max_dim), cv2.INTER_AREA)
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
        self.zoom_x1, self.zoom_y1, self.zoom_x2, self.zoom_y2 = 0,0,self.image_width,self.image_height
        self.flag_zoom = -1
        self.zoom_idx = 1

        self.zoom_n_increments = kwargs.get("zoom_steps", 10)
        self.zoom_step_x, self.zoom_step_y = int(image_width/self.zoom_n_increments), int(image_height/self.zoom_n_increments)
        
        if self.flag_zoom_mode == "fixed":
            magnification = kwargs.get("magnification", 5)
            if magnification >= self.zoom_n_increments:
                magnification = self.zoom_n_increments - 1
            self.zoom_step_x, self.zoom_step_y = magnification * self.zoom_step_x , magnification * self.zoom_step_y
            
        if self.flag_mode == "interactive":
            if self.flag_tool == "rectangle" or self.flag_tool == "box":
                self.rect_list = []            
                self.rect_start = None
            elif self.flag_tool == "polygon" or self.flag_tool == "free":
                self.poly_list = []
                self.point_list = []
            self.line_thickness = _auto_line_thickness(self.image)
        else:
            self.line_thickness = None
            
        ## update from a previous call
        if kwargs.get("prev_attributes"):
            prev_attr = kwargs.get("prev_attributes")
            prev_attr = {i:prev_attr[i] for i in prev_attr if i not in ["canvas_copy", "canvas", "image_copy","image"]}
            self.__dict__.update(prev_attr)
            if hasattr(self, "rect_list"):
                for (rx1, ry1, rx2, ry2) in self.rect_list:
                            cv2.rectangle(self.image_copy, (rx1,ry1), (rx2,ry2), colours.green, self.line_thickness)
            if hasattr(self, "poly_list"):
                for poly in self.poly_list:
                    cv2.polylines(self.image_copy, np.array([poly]), False, colours.green, self.line_thickness)
            self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2,]
            self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
            self.canvas_copy = copy.deepcopy(self.canvas)

        ## show canvas
        self.done = False
        cv2.namedWindow(self.window_name, self.window_aspect)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        cv2.resizeWindow(self.window_name, self.canvas_width, self.canvas_height)
        cv2.imshow(self.window_name, self.canvas)

        ## window control
        if self.window_control=="internal":
            if cv2.waitKey() == 13:
                self.done = True
                cv2.destroyAllWindows()
                if self.flag_tool == "polygon" or self.flag_tool == "free":
                    if len(self.point_list)>2:
                        self.point_list.append(self.point_list[0])
                        self.poly_list.append(self.point_list)
            elif cv2.waitKey() == 27:
                cv2.destroyAllWindows()
                sys.exit("Esc: exit phenopype process")

    def _on_mouse(self, event, x, y, flags, params):
        """Helper functon that defines mouse callback for image_viewer.
        """
        if event == cv2.EVENT_MOUSEWHEEL and flags > 0:
            if self.zoom_idx < self.zoom_n_increments:
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
        if self.flag_mode == "interactive":
            if self.flag_tool == "rectangle" or self.flag_tool == "box":
                if event == cv2.EVENT_LBUTTONDOWN: ## and (flags & cv2.EVENT_FLAG_CTRLKEY)
                    self.rect_start = x, y
                    self.canvas_copy = copy.deepcopy(self.canvas)
                if event == cv2.EVENT_LBUTTONUP: ## and (flags & cv2.EVENT_FLAG_CTRLKEY)
                    self.rect_start = None
                    self.rect_list.append([
                            int(self.zoom_x1 + (self.global_fx * self.rect_minpos[0])), int(self.zoom_y1 + (self.global_fy * self.rect_minpos[1])),
                            int(self.zoom_x1 + (self.global_fx * self.rect_maxpos[0])), int(self.zoom_y1 + (self.global_fy * self.rect_maxpos[1]))])
                    for (rx1, ry1, rx2, ry2) in self.rect_list:
                        cv2.rectangle(self.image_copy, (rx1,ry1), (rx2,ry2), colours.green, self.line_thickness)
                    self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                    self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                    self.canvas_copy = copy.deepcopy(self.canvas)
                    cv2.imshow(self.window_name, self.canvas)
                if event == cv2.EVENT_RBUTTONDOWN:
                    if len(self.rect_list)>0:
                        self.rect_list = self.rect_list[:-1]
                        self.image_copy = copy.deepcopy(self.image)
                        for (rx1, ry1, rx2, ry2) in self.rect_list:
                            cv2.rectangle(self.image_copy, (rx1,ry1), (rx2,ry2), colours.green, self.line_thickness)
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
                                      colours.red, max(2, _auto_line_thickness(self.canvas)))
                        cv2.imshow(self.window_name, self.canvas)
            elif self.flag_tool == "polygon" or self.flag_tool == "free":
                if event == cv2.EVENT_MOUSEMOVE:
                    self.coords_original = int(self.zoom_x1+(x * self.global_fx)), int(self.zoom_y1+(y * self.global_fy))
                    if len(self.point_list) > 0:
                        self.coords_prev = int((self.point_list[-1][0]-self.zoom_x1)/self.global_fx), int((self.point_list[-1][1]-self.zoom_y1)//self.global_fy)
                        self.canvas = copy.deepcopy(self.canvas_copy)
                        cv2.line(self.canvas, self.coords_prev, (x,y), colours.blue, self.line_thickness)
                    cv2.imshow(self.window_name, self.canvas)
                if event == cv2.EVENT_LBUTTONDOWN: ## and (flags & cv2.EVENT_FLAG_CTRLKEY)
                    self.coords_original = int(self.zoom_x1+(x * self.global_fx)), int(self.zoom_y1+(y * self.global_fy))
                    self.point_list.append(self.coords_original)
                    cv2.polylines(self.image_copy, np.array([self.point_list]), False, colours.green, self.line_thickness)
                    if len(self.poly_list)>0:
                        for poly in self.poly_list:
                            cv2.polylines(self.image_copy, np.array([poly]), False, colours.green, self.line_thickness)
                    self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                    self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                    self.canvas_copy = copy.deepcopy(self.canvas)
                    cv2.imshow(self.window_name, self.canvas)
                if event == cv2.EVENT_RBUTTONDOWN:
                    if len(self.point_list)>0:
                        self.point_list = self.point_list[:-1]
                        self.image_copy = copy.deepcopy(self.image)
                        cv2.polylines(self.image_copy, np.array([self.point_list]), False, colours.green, self.line_thickness)
                        for poly in self.poly_list:
                            cv2.polylines(self.image_copy, np.array([poly]), False, colours.green, self.line_thickness)
                        self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                        self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                        self.canvas_copy = copy.deepcopy(self.canvas)
                        cv2.imshow(self.window_name, self.canvas)
                    elif len(self.point_list) == 0 and len(self.poly_list)>0:
                        self.poly_list = self.poly_list[:-1]
                        self.image_copy = copy.deepcopy(self.image)
                        for poly in self.poly_list:
                            cv2.polylines(self.image_copy, np.array([poly]), False, colours.green, self.line_thickness)
                        self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                        self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                        self.canvas_copy = copy.deepcopy(self.canvas)
                        cv2.imshow(self.window_name, self.canvas)
                if flags == cv2.EVENT_FLAG_CTRLKEY and len(self.point_list)>2:
                    self.point_list.append(self.point_list[0])
                    self.poly_list.append(self.point_list)
                    self.point_list = []
                    self.image_copy = copy.deepcopy(self.image)
                    if len(self.poly_list)>0:
                        for poly in self.poly_list:
                            cv2.polylines(self.image_copy, np.array([poly]), False, colours.green, self.line_thickness)
                    self.canvas = self.image_copy[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
                    self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
                    self.canvas_copy = copy.deepcopy(self.canvas)

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
            x1, x2, y1, y2 = self.zoom_x1 + left_padding, self.zoom_x2 - right_padding, self.zoom_y1 + top_padding, self.zoom_y2 - bottom_padding
        if self.flag_zoom < 0:
            x1, x2, y1, y2 = self.zoom_x1 - left_padding, self.zoom_x2 + right_padding, self.zoom_y1 - top_padding, self.zoom_y2 + bottom_padding
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
        self.canvas = self.image_copy[y1:y2,x1:x2]

        self.global_fx, self.global_fy = self.canvas_fx * ((self.zoom_x2-self.zoom_x1)/self.image_width), self.canvas_fy * ((self.zoom_y2-self.zoom_y1)/self.image_height)
        self.canvas = cv2.resize(self.canvas, (self.canvas_width, self.canvas_height),interpolation = cv2.INTER_LINEAR)
        self.canvas_copy = copy.deepcopy(self.canvas)