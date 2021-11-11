#%% modules

import cv2, copy, os, sys, warnings
import numpy as np
import pandas as pd
from dataclasses import make_dataclass

import time
from timeit import default_timer as timer
import ruamel.yaml
from ruamel.yaml.constructor import SafeConstructor

from datetime import datetime
from math import cos
from pathlib import Path
from PIL import Image
from stat import S_IWRITE
from ruamel.yaml import YAML
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# from phenopype.settings import colours, confirm_options, pype_config_template_list, opencv_interpolation_flags
# from phenopype.settings import flag_verbose, opencv_window_flags, AttrDict

import phenopype.settings as settings

## capture yaml output - temp
from contextlib import redirect_stdout
import io

from phenopype import _config
from phenopype import main


#%% settings

Image.MAX_IMAGE_PIXELS = 999999999

#%% classes

# @_image_viewer_settings
class _ImageViewer:
    def __init__(self, 
                 image, 
                 tool=None,
                 window_aspect='normal', 
                 window_control='internal', 
                 zoom_magnification=0.5, 
                 zoom_mode='continuous', 
                 zoom_steps=20,
                 **kwargs):
        
        """
        Low level interactive image function.
        
        Future versions of phenopype will feature a more clean and userfriendly
        code structure.
        
        Parameters
        ----------

        """
        ## kwargs (FUTURE => use dictionary from config)
        
        if kwargs.get("window_max_dim"):
            window_max_dim = kwargs.get("window_max_dim")
        elif hasattr(main, "window_max_dim"):
            window_max_dim = main.window_max_dim
        else:
            window_max_dim = _config.window_max_dim
        
        ## args
        self.window_aspect = window_aspect
        self.window_control = window_control
        self.zoom_magnification = zoom_magnification
        self.zoom_mod = zoom_mode
        self.zoom_steps = zoom_steps

        self.flag_text_label = kwargs.get("flag_text_label", False)
        self.overlay_blend = kwargs.get("overlay_blend", 0.5)
        self.overlay_line_width = kwargs.get("overlay_line_width", 1)

        self.__dict__.update(kwargs)

        ## set class arguments
        self.tool = tool
        self.flag_zoom_mode = zoom_mode       
        self.zoom_magnification = zoom_magnification
        self.zoom_n_steps = zoom_steps
        self.wait_time = 100        
        self.window_name = "phenopype"
                
        ## needs cleaning
        self.flags = settings.AttrDict({"passive":False})
        self.flags.passive = kwargs.get("passive", False)
        
        # =============================================================================
        # initialize variables
        # =============================================================================

        ## image
        self.image = copy.deepcopy(image)
        self.image_width, self.image_height = self.image.shape[1], self.image.shape[0]
        
        ## binary image (for blending)
        if "contours" in kwargs and self.tool == "draw":
            
            ## get contours to create colour mask
            contours = kwargs.get("contours")

            ## coerce to multi channel image for colour mask
            if len(self.image.shape) == 2:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            
            ## create binary overlay
            self.image_bin = np.zeros(self.image.shape[0:2], dtype=np.uint8)
            
            ## draw contours onto overlay
            for contour in contours:
                cv2.drawContours(
                    image=self.image_bin,
                    contours=[contour],
                    contourIdx=0,
                    thickness=-1,
                    color=255,
                    maxLevel=3,
                    offset=(0,0),
                    )
                                          
        ## get canvas dimensions
        if self.image_height > window_max_dim or self.image_width > window_max_dim:
            if self.image_width >= self.image_height:
                self.canvas_width, self.canvas_height = window_max_dim, int(
                    (window_max_dim / self.image_width) * self.image_height)
            elif self.image_height > self.image_width:
                self.canvas_width, self.canvas_height = int(
                    (window_max_dim / self.image_height) * self.image_width), window_max_dim
        else:
            self.canvas_width, self.canvas_height = self.image_width, self.image_height
            
        ## canvas resize factor
        self.canvas_fx, self.canvas_fy = (
            self.image_width / self.canvas_width,
            self.image_height / self.canvas_height,
        )
        self.global_fx, self.global_fy = self.canvas_fx, self.canvas_fy

        ## zoom config
        self.zoom_x1, self.zoom_y1, self.zoom_x2, self.zoom_y2 = (
            0,
            0,
            self.image_width,
            self.image_height,
        )
        self.flag_zoom, self.zoom_idx = -1, 1
        self.zoom_step_x, self.zoom_step_y = (
            int(self.image_width / self.zoom_n_steps),
            int(self.image_height / self.zoom_n_steps),
        )
        if self.flag_zoom_mode == "fixed":
            mag = int(self.zoom_magnification * self.zoom_n_steps)
            self.zoom_step_x, self.zoom_step_y = (
                mag * self.zoom_step_x,
                mag * self.zoom_step_y,
            )

        # =============================================================================
        # configure tools
        # =============================================================================
        
        if self.tool:
            
            ## collect interactions and set flags
            self.points, self.point_list, self.coord_list = [], [], []
            self.rect_start, self.drawing = None, False
            
            ## line properties
            self.line_colour = settings.colours[kwargs.get("line_colour", "green")]
            self.line_width = kwargs.get("line_width", _auto_line_width(image))
            self.line_width_orig = copy.deepcopy(self.line_width)
            
            ## point properties
            self.point_size = kwargs.get("point_size", _auto_point_size(image))
            self.point_colour = kwargs.get("point_colour", "red")
            self.text_size = kwargs.get("label_size", _auto_text_size(image))
            self.text_width = kwargs.get("label_size", _auto_text_width(image))
            self.label_colour = kwargs.get("label_colour", "black")
            
            ## for contour edit tool
            self.left_colour = kwargs.get("left_colour", "green")
            self.right_colour = kwargs.get("right_colour", "red")
            
            ## initialize comment tool
            self.field = kwargs.get("field", "")
            self.entry = ""
                            
        # =============================================================================
        # update self with parameters from previous instance
        # =============================================================================              
        
        ## update from previous call
        if kwargs.get("ImageViewer_previous"):
            prev_attr = kwargs.get("ImageViewer_previous").__dict__
            prev_attr = {
                i: prev_attr[i] for i in prev_attr 
                
                ## don't update arrays or provided kwargs
                if i not in ["canvas_copy", "canvas", "image_copy", "image", "image_bin"] + list(kwargs.keys())
            }
            
            self.__dict__.update(copy.deepcopy(prev_attr))
            

        # =============================================================================
        # generate canvas
        # =============================================================================
        
        ## initialize canvas
        self._canvas_renew()
        if self.tool in ["rectangle", "polygon", "polyline", "draw"]:
            self._canvas_draw(tool="line", coord_list=self.coord_list)
        if self.tool in ["point"]:
            self._canvas_draw(tool="point", coord_list=self.points)
        if self.tool in ["draw"]:
            self._canvas_draw(tool="line_bin", coord_list=self.point_list)
            self._canvas_blend()
            self._canvas_add_lines()
        self._canvas_mount()

        ## local control vars
        self.done = False
        self.finished = False
        _config.window_close = False
        
        # =============================================================================
        # window control
        # =============================================================================
        
        if self.flags.passive == True:
            
            self.done = True
            self.finished = True
            
        else:
            
            cv2.namedWindow(self.window_name, settings.opencv_window_flags[window_aspect])
            cv2.startWindowThread() 
            cv2.setMouseCallback(self.window_name, self._on_mouse_plain)
            cv2.resizeWindow(self.window_name, self.canvas_width, self.canvas_height)
            cv2.imshow(self.window_name, self.canvas)
            self.keypress = None
        
            while not any([self.finished, self.done]):
                cv2.imshow(self.window_name, self.canvas)
                if self.flags.passive == False:
                    
                    ## comment tool
                    if self.tool == "comment":
                        self.keypress = cv2.waitKey(1)
                        self._comment_tool()
                    else:
                        self.keypress = cv2.waitKey(500)                 
    
                    ## Enter = close window and redo
                    if self.keypress == 13:
                        ## close unfinished polygon and append to polygon list
                        if self.tool:
                            if len(self.points) > 2 and not self.tool in ["point"]:
                                self.points.append(self.points[0])
                                self.coord_list.append(self.points)
                        self.done = True
                        cv2.destroyAllWindows()
                        
                    ## Ctrl + Enter = close window and move on
                    elif self.keypress == 10:
                        self.done = True
                        self.finished = True
                        cv2.destroyAllWindows()
                        print("finish")
                        
                    ## Esc = close window and terminate
                    elif self.keypress == 27:
                        cv2.destroyAllWindows()
                        sys.exit("\n\nTERMINATE (by user)")
                        
                    ## Ctrl + z = undo
                    elif self.keypress == 26 and self.tool == "draw":
                        self.point_list = self.point_list[:-1]
                        self._canvas_renew()
                        self._canvas_draw(tool="line_bin", coord_list=self.point_list)
                        self._canvas_blend()
                        self._canvas_add_lines()
                        self._canvas_mount()
                        
                    ## external window close
                    elif _config.window_close:
                        self.done = True
                        cv2.destroyAllWindows()
                    
                    
    def _comment_tool(self):
                
        if self.keypress > 0 and not self.keypress in [8, 13, 27]:
            self.entry = self.entry + chr(self.keypress)
        elif self.keypress == 8:
            self.entry = self.entry[0 : len(self.entry) - 1]

        self.canvas = copy.deepcopy(self.canvas_copy)
        cv2.putText(
            self.canvas,
            "Enter " + self.field + ": " + self.entry,
            (int(self.canvas.shape[0] // 10), int(self.canvas.shape[1] / 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_size,
            settings.colours[self.label_colour],
            self.text_width,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, self.canvas)
                    
    def _on_mouse_plain(self, event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEWHEEL and not self.keypress == 9:
            self.keypress = None
            if flags > 0:
                if self.zoom_idx < self.zoom_n_steps:
                    self.flag_zoom = 1
                    self.zoom_idx += 1
                    if self.flag_zoom_mode == "continuous" or (
                        self.flag_zoom_mode == "fixed" and self.zoom_idx == 2
                    ):
                        self._zoom_fun(x, y)
            if flags < 0:
                if self.zoom_idx > 1:
                    self.flag_zoom = -1
                    self.zoom_idx -= 1
                    if self.flag_zoom_mode == "continuous" or (
                        self.flag_zoom_mode == "fixed" and self.zoom_idx == 1
                    ):
                        self._zoom_fun(x, y)
            self.x, self.y = x, y
            cv2.imshow(self.window_name, self.canvas)

        if self.tool:
            if self.tool == "draw":
                self._on_mouse_draw(event, x, y, flags)
            elif self.tool == "point":
                self._on_mouse_point(event, x, y)
            elif self.tool == "polygon":
                self._on_mouse_polygon(event, x, y, flags)
            elif self.tool == "polyline" or self.tool == "polylines":
                self._on_mouse_polygon(event, x, y, flags, polyline=True)
            elif self.tool == "rectangle":
                self._on_mouse_rectangle(event, x, y, flags)
            elif self.tool == "reference":
                self._on_mouse_polygon(event, x, y, flags, reference=True)
            elif self.tool == "template":
                self._on_mouse_rectangle(event, x, y, flags, template=True)

                
                
    def _on_mouse_point(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
        
            ## convert cursor coords from zoomed canvas to original coordinate space
            self._zoom_coords_orig(x,y)
            
            ## append points to point list
            self.points.append(self.coords_original)
                       
            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(tool="point", coord_list=self.points)
            self._canvas_mount()
            
        if event == cv2.EVENT_RBUTTONDOWN:
            
            ## remove points from list, if any are left
            if len(self.points) > 0:
                self.points = self.points[:-1]
                
            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(tool="point", coord_list=self.points)
            self._canvas_mount()
                

    def _on_mouse_polygon(self, event, x, y, flags, **kwargs):

        ## kwargs
        polyline = kwargs.get("polyline", False)
        reference = kwargs.get("reference", False)
        flag_draw = kwargs.get("draw", False)

        if event == cv2.EVENT_MOUSEMOVE:
            if (reference or flag_draw) and self.tool == "line" and len(self.points) == 2:
                return
            
            ## draw line between current cursor coords and last polygon node
            if len(self.points) > 0:
                self.coords_prev = (
                    int((self.points[-1][0] - self.zoom_x1) / self.global_fx),
                    int((self.points[-1][1] - self.zoom_y1) // self.global_fy),
                )
                self.canvas = copy.deepcopy(self.canvas_copy)
                cv2.line(
                    self.canvas,
                    self.coords_prev,
                    (x, y),
                    self.line_colour,
                    self.line_width,
                )
                
            ## if in reference mode, don't connect
            elif (reference or flag_draw) and self.tool == "line" and len(self.points) > 2:
                pass
            
            ## pump updates
            cv2.imshow(self.window_name, self.canvas)
            
        if event == cv2.EVENT_LBUTTONDOWN:  
        
            ## skip if in reference mode
            if reference and len(self.points) == 2:
                print("already two points selected")
                return
            
            ## convert cursor coords from zoomed canvas to original coordinate space
            self._zoom_coords_orig(x,y)
            
            ## append points to point list
            self.points.append(self.coords_original)
            
            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="line", coord_list=self.coord_list + [self.points])
            self._canvas_mount()
            
            ## if in reference mode, append to ref coords
            if reference and len(self.points) == 2:
                print("Reference set")
                self.reference_coords = self.points
                                                
        if event == cv2.EVENT_RBUTTONDOWN:
            
            ## remove points and update canvas
            if len(self.points) > 0:
                self.points = self.points[:-1]
            else:
                self.coord_list = self.coord_list[:-1]
                
            ## apply tool and refresh canvas
            print("remove")
            self._canvas_renew()
            self._canvas_draw(
                tool="line", coord_list=self.coord_list + [self.points])
            self._canvas_mount()

        if flags == cv2.EVENT_FLAG_CTRLKEY and len(self.points) > 2:

            ## close polygon
            if not polyline:
                self.points.append(self.points[0])
                
            ## add current points to polygon and empyt point list
            print("poly")
            self.coord_list.append(self.points)
            self.points = []

            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="line", coord_list=self.coord_list + [self.points])
            self._canvas_mount()


    def _on_mouse_rectangle(self, event, x, y, flags, **kwargs):
        
        ## kwargs
        template = kwargs.get("template", False)
        
        if event == cv2.EVENT_LBUTTONDOWN:  
            
            ## end after one set of points if creating a template
            if template == True and len(self.coord_list) == 1:
                return
            
            ## start drawing temporary rectangle 
            self.rect_start = x, y
            self.canvas_copy = copy.deepcopy(self.canvas)
            
        if event == cv2.EVENT_LBUTTONUP:
            
            ## end after one set of points if creating a template
            if template == True and len(self.coord_list) == 1:
                print("Template selected")
                return
            
            ## end drawing temporary rectangle
            self.rect_start = None

            ## convert rectangle to polygon coords
            self.rect = [
                int(self.zoom_x1 + (self.global_fx * self.rect_minpos[0])),
                int(self.zoom_y1 + (self.global_fy * self.rect_minpos[1])),
                int(self.zoom_x1 + (self.global_fx * self.rect_maxpos[0])),
                int(self.zoom_y1 + (self.global_fy * self.rect_maxpos[1])),
                ]
            self.coord_list.append(
                [
                    (self.rect[0], self.rect[1]), 
                    (self.rect[2], self.rect[1]),
                    (self.rect[2], self.rect[3]),
                    (self.rect[0], self.rect[3]),
                    (self.rect[0], self.rect[1]),
                    ]
            )           
            
            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="line", coord_list=self.coord_list)
            self._canvas_mount(refresh=False)
            
        if event == cv2.EVENT_RBUTTONDOWN:
            
            ## remove polygons and update canvas
            if len(self.coord_list) > 0:
                self.coord_list = self.coord_list[:-1]
                
                ## apply tool and refresh canvas
                self._canvas_renew()
                self._canvas_draw(
                    tool="line", coord_list=self.coord_list)
                self._canvas_mount()

                
        ## draw temporary rectangle
        elif self.rect_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:        
                self.canvas = copy.deepcopy(self.canvas_copy)
                self.rect_minpos = min(self.rect_start[0], x), min(self.rect_start[1], y)
                self.rect_maxpos = max(self.rect_start[0], x), max(self.rect_start[1], y)
                cv2.rectangle(
                    self.canvas,
                    self.rect_minpos,
                    self.rect_maxpos,
                    self.line_colour,
                    self.line_width,
                )
                cv2.imshow(self.window_name, self.canvas)
                
                
    def _on_mouse_draw(self, event, x, y, flags):     

        ## set colour - left/right mouse button use different colours
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.colour_current_bin = 255
                self.colour_current = settings.colours[self.left_colour]
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.colour_current_bin = 0
                self.colour_current = settings.colours[self.right_colour]
            
            ## start drawing and use current coords as start point
            self.canvas = copy.deepcopy(self.canvas_copy)
            
            ## convert cursor coords from zoomed canvas to original coordinate space
            self.ix,self.iy=x,y
            self.coords_original_i = (
                int(self.zoom_x1 + (self.ix * self.global_fx)),
                int(self.zoom_y1 + (self.iy * self.global_fy)),
            )
            self.points.append(self.coords_original_i)
            self.drawing=True

        ## finish drawing and update image_copy
        if event==cv2.EVENT_LBUTTONUP or event==cv2.EVENT_RBUTTONUP:
            self.drawing=False
            self.canvas = copy.deepcopy(self.canvas_copy)
            self.point_list.append([
                self.points,
                self.colour_current_bin, 
                int(self.line_width*self.global_fx),
                ])
            self.points = []
            
            ## draw all segments
            self._canvas_renew()
            self._canvas_draw(tool="line_bin", coord_list=self.point_list)
            self._canvas_blend()
            self._canvas_add_lines()
            self._canvas_mount()
                
        ## drawing mode
        elif self.drawing:

            ## convert cursor coords from zoomed canvas to original coordinate space
            self._zoom_coords_orig(x,y)
            
            ## add points, colour, and line width to point list
            self.points.append(self.coords_original)
            
            ## draw onto canvas for immediate feedback
            cv2.line(self.canvas,(self.ix,self.iy),(x,y), 
                     self.colour_current, self.line_width) 
            self.ix,self.iy = x,y
            cv2.imshow(self.window_name, self.canvas)  
                        
        if self.keypress == 9 and event == cv2.EVENT_MOUSEWHEEL:
            if flags > 1:
                self.line_width_orig += 1
            if flags < 1 and self.line_width_orig > 1:
                self.line_width_orig -= 1

            self.canvas = copy.deepcopy(self.canvas_copy)
            self.line_width = int(
                self.line_width_orig / ((self.zoom_x2 - self.zoom_x1) / self.image_width))
            cv2.line(self.canvas, (x, y), (x, y),
                     settings.colours["black"], self.line_width)
            cv2.line(self.canvas, (x, y), (x, y),
                     settings.colours["white"], max(self.line_width-5, 1))
            cv2.imshow(self.window_name, self.canvas)
            
    
    def _canvas_add_lines(self):
        
        _ , self.contours, self.hierarchies = cv2.findContours(
            image=self.image_bin_copy,
            mode=cv2.RETR_CCOMP,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        
        for contour in self.contours:
            cv2.drawContours(
                image=self.image_copy,
                contours=[contour],
                contourIdx=0,
                thickness=self.overlay_line_width,
                color=settings.colours[self.left_colour],
                maxLevel=3,
                offset=None,
            )   
            
            
    def _canvas_blend(self):
        
        ## create coloured overlay from binary image
        self.colour_mask = copy.deepcopy(self.image_bin_copy)
        self.colour_mask = cv2.cvtColor(self.colour_mask, cv2.COLOR_GRAY2BGR)
        self.colour_mask[self.image_bin_copy == 0] = settings.colours[self.right_colour]
        self.colour_mask[self.image_bin_copy == 255] = settings.colours[self.left_colour]

        ## blend two canvas layers
        self.image_copy = cv2.addWeighted(self.image_copy,
                                          1 - self.overlay_blend,
                                          self.colour_mask,
                                          self.overlay_blend,
                                          0)           
        
        
    def _canvas_draw(self, tool, coord_list):
                              
        ## apply coords to tool and draw on canvas
        for idx, coords in enumerate(coord_list):
            if len(coords)==0:
                continue
            if tool == "line":
                cv2.polylines(
                    self.image_copy,
                    np.array([coords]),
                    False,
                    self.line_colour,
                    self.line_width,
                )
            elif tool == "line_bin":
                cv2.polylines(
                    self.image_bin_copy,
                    np.array([coords[0]]),
                    False,
                    coords[1],
                    coords[2],
                )
                
                
    def _canvas_mount(self, refresh=True):
              
        ## pass zoomed part of original image to canvas
        self.canvas = self.image_copy[
            self.zoom_y1 : self.zoom_y2, self.zoom_x1 : self.zoom_x2
        ]
        
        ## resize canvas to fit window
        self.canvas = cv2.resize(
            self.canvas,
            (self.canvas_width, self.canvas_height),
            interpolation=cv2.INTER_LINEAR,
        )
        
        ## copy canvas for mousedrag refresh
        self.canvas_copy = copy.deepcopy(self.canvas)
        
        ## refresh canvas
        if refresh:
            cv2.imshow(self.window_name, self.canvas)


    def _canvas_renew(self):

        ## pull copy from original image
        self.image_copy = copy.deepcopy(self.image)
        if self.tool=="draw":
            self.image_bin_copy = copy.deepcopy(self.image_bin)


    def _zoom_fun(self, x, y):
        """
        Helper function for image_viewer. Takes current xy coordinates and 
        zooms in within a rectangle around mouse coordinates while transforming 
        current cursor coordinates back to original coordinate space
        
        """
        if y <= 0:
            y = 1
        if x <= 0:
            x = 1

        x_prop, y_prop = x / self.canvas_width, y / self.canvas_height
        left_padding, right_padding = (
            int(round(x_prop * self.zoom_step_x)),
            int(round((1 - x_prop) * self.zoom_step_x)),
        )
        top_padding, bottom_padding = (
            int(round(y_prop * self.zoom_step_y)),
            int(round((1 - y_prop) * self.zoom_step_y)),
        )

        if self.flag_zoom > 0:
            x1, x2 = self.zoom_x1 + left_padding, self.zoom_x2 - right_padding
            y1, y2 = self.zoom_y1 + top_padding, self.zoom_y2 - bottom_padding
        if self.flag_zoom < 0:
            x1, x2 = (
                self.zoom_x1 - left_padding,
                self.zoom_x2 + right_padding,
            )
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

        ## failsafe when zooming out, sets zoom-coords to image coords
        if self.zoom_idx == 1:
            x1,x2,y1,y2 = 0, self.image_width, 0, self.image_height

        ## zoom coords
        self.zoom_x1, self.zoom_x2, self.zoom_y1, self.zoom_y2 = x1, x2, y1, y2

        ## global magnification factor
        self.global_fx = self.canvas_fx * (
            (self.zoom_x2 - self.zoom_x1) / self.image_width
        )
        self.global_fy = self.canvas_fy * (
            (self.zoom_y2 - self.zoom_y1) / self.image_height
        )

        ## update canvas
        self._canvas_mount(refresh=False)
        
        ## adjust brush size
        if self.tool == "draw":
            self.line_width = int(self.line_width_orig / ((self.zoom_x2 - self.zoom_x1) / self.image_width))


    def _zoom_coords_orig(self, x, y):
        self.coords_original = (
                int(self.zoom_x1 + (x * self.global_fx)),
                int(self.zoom_y1 + (y * self.global_fy)),
            )
        
        

class _YamlFileMonitor:
    def __init__(self, filepath, delay=500):

        ## file, location and event action
        self.dirpath = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)
        self.filepath = filepath
        self.event_handler = PatternMatchingEventHandler(patterns=["*/" + self.filename])
        self.event_handler.on_any_event = self._on_update

        ## intitialize
        self.content = _load_yaml(self.filepath)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.dirpath, recursive=False)
        self.observer.start()
        self.delay = delay
        self.time_start = None
        self.time_diff = 10
        
    def _on_update(self, event):
        if not self.time_start.__class__.__name__ == "NoneType":
            self.time_end = timer()
            self.time_diff = self.time_end - self.time_start
        
        if self.time_diff > 1:
            self.content = _load_yaml(self.filepath)
            _config.window_close,_config.pype_restart = True, True
            cv2.destroyWindow("phenopype")
            cv2.waitKey(self.delay)
        else:
            pass
        
        self.time_start = timer()
        
    def _stop(self):
        self.observer.stop()
        self.observer.join()



class _DummyClass:
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)

#%% functions


def _auto_line_width(image, **kwargs):
    factor = kwargs.get("factor", 0.001)
    image_height, image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) / 2
    line_width = max(int(factor * image_diagonal), 1)

    return line_width


def _auto_point_size(image, **kwargs):
    factor = kwargs.get("factor", 0.002)
    image_height, image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) / 2
    point_size = max(int(factor * image_diagonal), 1)

    return point_size


def _auto_text_width(image, **kwargs):
    factor = kwargs.get("factor", 0.0005)
    image_height, image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) / 2
    text_width = max(int(factor * image_diagonal), 1)

    return text_width


def _auto_text_size(image, **kwargs):
    factor = kwargs.get("factor", 0.00025)
    image_height, image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) / 2
    text_size = max(int(factor * image_diagonal), 1)

    return text_size


def _convert_arr_tup_list(arr_list):
    
    if not arr_list.__class__.__name__ == "list":
        arr_list = [arr_list]
    
    tup_list = []
    for array in arr_list:
        point_list = []
        for point in array:
            point_list.append(tuple((int(point[0][0]),int(point[0][1]))))
        tup_list.append(point_list)
    return tup_list


def _convert_tup_list_arr(tup_list):
    array_list = []
    for points in tup_list:
        point_list = []
        for point in points:
            point_list.append([list(point)])
        array_list.append(np.asarray(point_list, dtype="int32"))
    return array_list


def _create_mask_bin(image, contours):
    mask_bin = np.zeros(image.shape[0:2], np.uint8)
    if contours[0].__class__.__name__ == "list":
        cv2.fillPoly(mask_bin, [np.array(contours, dtype=np.int32)], settings.colours["white"])
    elif contours[0].__class__.__name__ == "ndarray":
        for contour in contours:
            cv2.fillPoly(mask_bin, [np.array(contour, dtype=np.int32)], settings.colours["white"])
    return mask_bin


def _create_mask_bool(image, contours):
    mask_bin = _create_mask_bin(image, contours)
    return np.array(mask_bin, dtype=bool)


def _decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def _del_rw(action, name, exc):
    os.chmod(name, S_IWRITE)
    os.remove(name)
    
    
def _df_overwrite_checker(df, 
                          annotation, 
                          label,
                          flag_overwrite, 
                          flag_edit):

    if df.__class__.__name__ == "NoneType":
        df = pd.DataFrame()
        print("- creating new {}".format(annotation))
        return df, None
    elif df.__class__.__name__ == "DataFrame":
        if label in df["mask"].values:
            if flag_overwrite == False and flag_edit == False:
                print("- {} with label \"{}\" already created (edit/overwrite=False)".format(annotation,label))
                return None, None
            elif flag_overwrite == True and flag_edit == False:
                df = df.drop(df[df["mask"]==label].index)
                print("- creating {} (overwriting)".format(annotation))
                return df, None
            elif flag_edit == True:
                edit_params = df[df["mask"]==label].to_dict("records")[0]
                df = df.drop(df[df["mask"]==label].index)
                print("- creating {} (editing)".format(annotation))
                return df, edit_params  
        else:
            print("- creating another {}".format(annotation))
            return df, None    
    else:
        print("- wrong df supplied to edit {}".format(annotation))

        
    
    
def _equalize_histogram(image, detected_rect_mask, template):
    """Histogram equalization via interpolation, upscales the results from the detected reference card to the entire image.
    May become a standalone function at some point in the future. THIS STRONGLY DEPENDS ON THE QUALITY OF YOUR TEMPLATE.
    Mostly inspired by this SO question: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    More theory here: https://docs.opencv.org/master/d4/d1b/tutorial_histogram_equalization.html
    """
    detected_ravel = detected_rect_mask.ravel()
    template_ravel = template.ravel()

    detected_counts = np.bincount(detected_ravel, minlength=256)
    detected_quantiles = np.cumsum(detected_counts).astype(np.float64)
    detected_quantiles /= detected_quantiles[-1]

    template_values = np.arange(0, 256, 1, dtype=np.uint8)
    template_counts = np.bincount(template_ravel, minlength=256)
    template_quantiles = np.cumsum(template_counts).astype(np.float64)
    template_quantiles /= template_quantiles[-1]

    interp_template_values = np.interp(
        detected_quantiles, template_quantiles, template_values
    )
    interp_template_values = interp_template_values.astype(image.dtype)

    return interp_template_values[image]



def _file_walker(
    directory,
    filetypes=[],
    include=[],
    include_all=True,
    exclude=[],
    recursive=False,
    unique="path",
    **kwargs
):
    """
    
    Parameters
    ----------
    directory : str
        path to directory to search for files
    recursive: (optional): bool,
        "False" searches only current directory for valid files; "True" walks 
        through all subdirectories
    filetypes (optional): list of str
        single or multiple string patterns to target files with certain endings
    include (optional): list of str
        single or multiple string patterns to target certain files to include
    include_all (optional): bool,
        either all (True) or any (False) of the provided keywords have to match
    exclude (optional): list of str
        single or multiple string patterns to target certain files to exclude - can overrule "include"
    unique (optional): str (default: "filepath")
        how should unique files be identified: "filepath" or "filename". "filepath" is useful, for example, 
        if identically named files exist in different subfolders (folder structure will be collapsed and goes into the filename),
        whereas filename will ignore all those files after their first occurrence.

    Returns
    -------
    None.

    """
    ## kwargs
    pype_mode = kwargs.get("pype_mode", False)
    if not filetypes.__class__.__name__ == "list":
        filetypes = [filetypes]
    if not include.__class__.__name__ == "list":
        include = [include]
    if not exclude.__class__.__name__ == "list":
        exclude = [exclude]
    flag_include_all = include_all
    flag_recursive = recursive
    flag_unique = unique
    
    ## find files
    filepaths1, filepaths2, filepaths3, filepaths4 = [], [], [], []
    if flag_recursive == True:
        for root, dirs, files in os.walk(directory):
            for file in os.listdir(root):
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):
                    filepaths1.append(filepath)
    else:
        for file in os.listdir(directory):
            filepath = os.path.join(directory, file)
            if os.path.isfile(filepath):
                filepaths1.append(filepath)

    ## file endings
    if len(filetypes) > 0:
        for filepath in filepaths1:
            if filepath.endswith(tuple(filetypes)):
                filepaths2.append(filepath)
    elif len(filetypes) == 0:
        filepaths2 = filepaths1

    ## include
    if len(include) > 0:
        for filepath in filepaths2:
            if flag_include_all:
                if all(inc in os.path.basename(filepath) for inc in include):
                    filepaths3.append(filepath)
            else:
                if any(inc in os.path.basename(filepath) for inc in include):
                    filepaths3.append(filepath)
    else:
        filepaths3 = filepaths2

    ## exclude
    if len(exclude) > 0:
        for filepath in filepaths3:
            if not any(exc in os.path.basename(filepath) for exc in exclude):
                filepaths4.append(filepath)
    else:
        filepaths4 = filepaths3

    ## check if files found
    filepaths = filepaths4
    if len(filepaths) == 0 and not pype_mode:
        sys.exit("No files found under the given location that match given criteria.")

    ## allow unique filenames filepath or by filename only
    filenames, unique_filename, unique, duplicate = [], [], [], []
    for filepath in filepaths:
        filenames.append(os.path.basename(filepath))
    if flag_unique in ["filepaths", "filepath", "path"]:
        for filename, filepath in zip(filenames, filepaths):
            if not filepath in unique:
                unique.append(filepath)
            else:
                duplicate.append(filepath)
    elif flag_unique in ["filenames", "filename", "name"]:
        for filename, filepath in zip(filenames, filepaths):
            if not filename in unique_filename:
                unique_filename.append(filename)
                unique.append(filepath)
            else:
                duplicate.append(filepath)

    return unique, duplicate



def _provide_pype_config(config=None, 
                         template=None,
                         name=None):
    
    ## kwargs and setup
    step_names = ["preprocessing", 
                  "segmentation", 
                  "measurement",  
                  "visualization", 
                  "export"]
    
    ## load exiisting config file
    if config.__class__.__name__ == "str":
        if not config.endswith(".yaml"):
            config = config + ".yaml"
        if os.path.isfile(config):
            config_name, config_path = os.path.basename(config), config
            print("Succesfully loaded existing pype config ({}) from:\n{} ".format((config_name),(config_path)))
            config = _load_yaml(config_path)
            return config
        else:    
            print("No config file found under: {}".format(config))
            return
        
    ## create config from template
    elif config.__class__.__name__ == "NoneType": 
        
        ## phenopype template 
        if isinstance(template, settings.Template):
            config_steps, config_path = template.processing_steps, "NA"
            template_name, template_path = template.name, template.path
            
        ## user template
        elif os.path.isfile(template):
            config_loaded = _load_yaml(template)
            if config_loaded.__class__.__name__ in ["dict", 'CommentedMap']:
                if "config_info" in config_loaded:
                    config_loaded.pop('config_info', None)
                    print("Removed existing \"config_info\" section")
                if "processing_steps" in config_loaded:
                    config_steps, config_path = config_loaded["processing_steps"], "NA"
                else:
                    print("Broken template - check for correct template structure")
                    return
            elif config_loaded.__class__.__name__ in ["list",'CommentedSeq'] and any(step in config_loaded[0] for step in step_names):
                config_steps, config_path = config_loaded, "NA"
            template_name, template_path = os.path.basename(template), os.path.abspath(template)

        else:
            print("Invalid template supplied - aborting.")
            return
    else:
        print("Could not load config or template - pease check input.")
        return 
    
    ## create config-layout
    if name.__class__.__name__ == "NoneType":
        config_name = "NA"
    elif name.__class__.__name__ == "str":
        config_name = "pype_config_" + name + ".yaml"
    config_info = {"config_name":config_name,
                   "config_path": config_path,
                   "template_name":template_name,
                   "template_path":template_path,
                   "date_created":datetime.today().strftime("%Y%m%d%H%M%S"),
                   "date_last_modified":None}
    config = {"config_info":config_info,
              "processing_steps":config_steps}
            
    ## return
    return config



def _get_circle_perimeter(center_x, center_y, radius):
    coordinate_list=[]
    for i in range(360):
        y = center_x + radius * cos(i)
        x = center_y + radius * cos(i)
        coordinate_list.append((int(x),int(y)))
    return coordinate_list



def _load_previous_annotation(annotation_previous, components, load_settings=True):
    ImageViewer_previous = {}    
    if load_settings:
        ImageViewer_previous.update(annotation_previous["settings"])
    for item in components:
        field, data = item
        ImageViewer_previous[data] = annotation_previous[field][data]

    return _DummyClass(ImageViewer_previous)


def _load_image_data(obj_input, path_and_type=True, resize=1):
    """
    Create a DataFreame with image information (e.g. dimensions).

    Parameters
    ----------
    obj_input: str or ndarray
        can be a path to an image stored on the harddrive OR an array already 
        loaded to Python.
    path_and_type: bool, optional
        return image path and filetype to image_data dictionary

    Returns
    -------
    image_data: dict
        contains image data (+meta data, if selected)

    """
    if obj_input.__class__.__name__ == "str":
        if os.path.isfile(obj_input):
            path = obj_input
        image = Image.open(path)
        width, height = image.size
        image.close()
        image_data = {
            "filename": os.path.split(obj_input)[1],
            "width": width,
            "height": height,
        }
        
        if path_and_type: 
            image_data.update({
                "filepath": obj_input,
                "filetype": os.path.splitext(obj_input)[1]})
            
    elif obj_input.__class__.__name__ == "ndarray":
        image = obj_input
        width, height = image.shape[0:2]
        image_data = {
            "filename": "unknown",
            "filepath": "unknown",
            "filetype": "ndarray",
            "width": width,
            "height": height,
        }
    else:
        warnings.warn("Not a valid image file - cannot read image data.")

    ## issue warnings for large images
    if width * height > 125000000:
        warnings.warn(
            "Large image - expect slow processing."
        )
    elif width * height > 250000000:
        warnings.warn(
            "Extremely large image - expect very slow processing \
                      and consider resizing."
        )

    ## return image data
    return image_data



def _drop_dict_entries(dictionary, drop=[]):
    
    new_dictionary = {}
    
    for key, value in dictionary.items():
        if not key in drop:
            new_dictionary[key] = value
            
    return new_dictionary



def _resize_image(image, factor=1, interpolation="cubic"):
    """
    Resize image by resize factor 

    Parameters
    ----------
    obj_input: array 
        image to be resized
    resize: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of 
        original size).
    interpolation: str, optional
        interpolation algorithm to use. check pp.settings.opencv_interpolation_flags
        and refer to https://docs.opencv.org/3.4.9/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

    Returns
    -------
    image : array or container
        resized image

    """
    
    ## method
    if factor == 1:
        pass
    else:
        image = cv2.resize(
            image, 
            (0, 0), 
            fx=1 * factor, 
            fy=1 * factor, 
            interpolation=settings.opencv_interpolation_flags[interpolation]
        )

    ## return results
    return image

def _load_yaml(filepath, typ="rt", pure=False, legacy=False):
        
    ## this can read phenopype < 2.0 style config yaml files
    if legacy==True:
        def _construct_yaml_map(self, node):
            data = []
            yield data
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=True)
                val = self.construct_object(value_node, deep=True)
                data.append((key, val))
    else:
        def _construct_yaml_map(self, node):
            data = self.yaml_base_dict_type()
            yield data
            value = self.construct_mapping(node)
            data.update(value) 
        
    SafeConstructor.add_constructor(u'tag:yaml.org,2002:map', _construct_yaml_map)
    yaml = YAML(typ=typ, pure=pure)

    if isinstance(filepath, (Path, str)):
        if Path(filepath).is_file():
            with open(filepath, "r") as file:
                return yaml.load(file)
            
        else:
            print("Cannot load config from specified filepath")
    else:
        print("Not a valid path - couldn't load yaml.")
        return
    
    
    
    isinstance(filepath, )

def _show_yaml(odict, ret=False, typ="rt"):
    
    yaml =  YAML(typ=typ)
    
    if ret:
        with io.StringIO() as buf, redirect_stdout(buf):
            yaml.dump(odict, sys.stdout)
            return buf.getvalue()
    else:
        yaml.dump(odict, sys.stdout)
    


def _save_yaml(dictionary, filepath, typ="rt"):
    yaml = YAML(typ=typ)
    with open(filepath, "w") as out:
        yaml.dump(dictionary, out)



def _update_settings(kwargs, local_settings, IV_settings=None):
    
    for key, value in kwargs.items():
        if key in settings._image_viewer_arg_list:
            if IV_settings:
                IV_settings[key] = value
            local_settings[key] = value



def _yaml_flow_style(dictionary):
   ret = ruamel.yaml.comments.CommentedMap(dictionary)
   ret.fa.set_flow_style()
   return ret   

# def _timestamp():
#     return datetime.today().strftime("%Y:%m:%d %H:%M:%S")


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
