#%% modules

import cv2, copy, os, sys, warnings
import json
import numpy as np
import pandas as pd
from dataclasses import make_dataclass
import string
import re
from _ctypes import PyObj_FromPtr
from colour import Color
from timeit import default_timer as timer
import ruamel.yaml
from ruamel.yaml.constructor import SafeConstructor
from ruamel.yaml import YAML

from math import cos
from pathlib import Path
from PIL import Image
from stat import S_IWRITE
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from contextlib import redirect_stdout
import io

from phenopype import _config
from phenopype import main
from phenopype import settings
from phenopype import utils

#%% classes


# @_image_viewer_settings
class _GUI:
    def __init__(
            self, 
            image, 
            tool=None,
            passive=False,
            wait_time=500,
            window_aspect='normal', 
            window_control='internal', 
            window_name="phenopype",
            zoom_magnification=0.5, 
            zoom_mode='continuous', 
            zoom_n_steps=20,
            **kwargs
            ):
        
        """
        Low level interactive image function.
        
        Future versions of phenopype will feature a more clean and userfriendly
        code structure.
        
        Parameters
        ----------

        """
        ## kwargs (improve: use dictionary from config)
        if kwargs.get("window_max_dim"):
            window_max_dim = kwargs.get("window_max_dim")
        elif hasattr(main, "window_max_dim"):
            if not main.window_max_dim == None:
                window_max_dim = main.window_max_dim
            else:
                window_max_dim = _config.window_max_dim
        else:
            window_max_dim = _config.window_max_dim
            
        
        self.__dict__.update(kwargs)

        ## basic settings
        self.tool = tool
        self.passive = passive
        self.label = kwargs.get("label", None)

        ## data collector
        self.data = {
            settings._comment_type: "",
            settings._contour_type:[],
            settings._coord_type:[],
            settings._coord_list_type:[],
            settings._sequence_type:[],
            }
        
        self.data.update(kwargs.get("data", {}))

        ## GUI settings 
        self.settings = make_dataclass(
            cls_name="settings", 
            fields=[
                
                ("show_label", bool, kwargs.get("show_label", False)),
                ("label_colour", tuple,  kwargs.get("label_colour", settings._default_label_colour)),
                ("label_size", int, kwargs.get("label_size", _auto_text_size(image))),
                ("label_width", int, kwargs.get("label_width", _auto_text_width(image))),
                
                ("line_colour", tuple,  kwargs.get("line_colour", settings._default_line_colour)),
                ("line_width", int, kwargs.get("line_width", _auto_line_width(image))),
                
                ("point_colour", tuple,  kwargs.get("point_colour", settings._default_point_colour)),
                ("point_size", int, kwargs.get("point_size", _auto_point_size(image))),
                
                ("overlay_blend", float, kwargs.get("overlay_blend", 0.2)),
                ("overlay_line_width", int, kwargs.get("overlay_line_width", 1)),
                ("overlay_colour_left", tuple, kwargs.get("overlay_colour_left", settings._default_overlay_left)),
                ("overlay_colour_right", tuple, kwargs.get("overlay_colour_right", settings._default_overlay_right)),
                
                ("zoom_mode", str, zoom_mode),
                ("zoom_magnification", float, zoom_magnification),
                ("zoom_n_steps", int, zoom_n_steps),
                
                ("wait_time", int, wait_time),
                
                ("window_aspect", str, window_aspect),
                ("window_control", str, window_control),
                ("window_max_dim", str, window_max_dim),
                ("window_name", str, window_name),
                
                ])   
                    
        self.locals = make_dataclass(
            cls_name="locals", 
            fields=[
                ]) 
        
        
        ## collect interactions and set flags
        self.line_width_orig = copy.deepcopy(self.settings.line_width)        


        self.flags = make_dataclass(
            cls_name="flags", 
            fields=[
                ("end", bool, False),
                ("end_pype", bool, False),
                ("drawing",bool, False),
                ("rect_start", tuple, None)
                ])   
        


        # =============================================================================
        # initialize variables
        # =============================================================================

        if not image.__class__.__name__ == "ndarray": 
            raise TypeError("GUI module did not receive array-type - aborting!")

        ## image
        self.image = copy.deepcopy(image)
        self.image_width, self.image_height = self.image.shape[1], self.image.shape[0]
        
        ## binary image (for blending)
        if self.tool == "draw":
            
            if len(self.data[settings._contour_type]) > 0:

                ## coerce to multi channel image for colour mask
                if len(self.image.shape) == 2:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
                
                ## create binary overlay
                self.image_bin = np.zeros(self.image.shape[0:2], dtype=np.uint8)
                
                ## draw contours onto overlay
                for contour in self.data[settings._contour_type]:
                    cv2.drawContours(
                        image=self.image_bin,
                        contours=[contour],
                        contourIdx=0,
                        thickness=-1,
                        color=255,
                        maxLevel=3,
                        offset=(0,0),
                        )
            else:
                raise AttributeError("Could not find contours to edit - check annotations.")
                                          
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
            int(self.image_width / self.settings.zoom_n_steps),
            int(self.image_height / self.settings.zoom_n_steps),
        )
        if self.settings.zoom_mode == "fixed":
            mag = int(self.settings.zoom_magnification * self.settings.zoom_n_steps)
            self.zoom_step_x, self.zoom_step_y = (
                mag * self.zoom_step_x,
                mag * self.zoom_step_y,
            )

        
        # ## update from previous call
        # if kwargs.get("ImageViewer_previous"):
        #     prev_attr = kwargs.get("ImageViewer_previous").__dict__
        #     prev_attr = {
        #         i: prev_attr[i] for i in prev_attr 
                
        #         ## don't update arrays or provided kwargs
        #         if i not in ["canvas_copy", "canvas", "image_copy", "image", "image_bin"] + list(kwargs.keys())
        #     }
            
        #     self.__dict__.update(copy.deepcopy(prev_attr))
            
        # =============================================================================
        # generate canvas
        # =============================================================================

        ## initialize canvas
        self._canvas_renew()
        if self.tool in ["rectangle", "polygon", "polyline", "draw"]:
            self._canvas_draw(tool="line", coord_list=self.data["polygons"])
        if self.tool in ["point"]:
            self._canvas_draw(tool="point", coord_list=self.data[settings._coord_type])
        if self.tool in ["draw"]:
            self._canvas_draw(tool="line_bin", coord_list=self.data["sequences"])
            self._canvas_blend()
            self._canvas_add_lines()
        self._canvas_mount()                                   


        ## local control vars
        _config.window_close = False
        
        
        # =============================================================================
        # window control
        # =============================================================================
        
        if self.passive == True:
            
            self.flags.end = True
            self.flags.end_pype= True
                       
        else:
            cv2.namedWindow(self.settings.window_name, settings.opencv_window_flags[window_aspect])
            cv2.startWindowThread() 
            cv2.setMouseCallback(self.settings.window_name, self._on_mouse_plain)
            cv2.resizeWindow(self.settings.window_name, self.canvas_width, self.canvas_height)
            cv2.imshow(self.settings.window_name, self.canvas)
            self.keypress = None
                    
            if self.settings.window_control == "internal":
                while not any([self.flags.end, self.flags.end_pype]):
                    if self.passive == False:
                        
                        ## comment tool
                        if self.tool == "comment":
                            self.keypress = cv2.waitKey(1)
                            self._comment_tool()
                        else:
                            self.keypress = cv2.waitKey(self.settings.wait_time)                 
        
                        ## Enter = close window and redo
                        if self.keypress == 13:
                            ## close unfinished polygon and append to polygon list
                            if self.tool:
                                if len(self.data[settings._coord_type]) > 2 and not self.tool in ["point"]:
                                    self.data[settings._coord_type].append(self.data[settings._coord_type][0])
                                    self.data["polygons"].append(self.data[settings._coord_type])
                            self.flags.end = True
                            cv2.destroyAllWindows()
                            
                        ## Ctrl + Enter = close window and move on
                        elif self.keypress == 10:
                            self.flags.end = True
                            self.flags.end_pype = True
                            cv2.destroyAllWindows()
                            
                        ## Esc = close window and terminate
                        elif self.keypress == 27:
                            cv2.destroyAllWindows()
                            sys.exit("\n\nTERMINATE (by user)")
                            
                        ## Ctrl + z = undo
                        elif self.keypress == 26 and self.tool == "draw":
                            self.data["sequences"] = self.data["sequences"][:-1]
                            self._canvas_renew()
                            self._canvas_draw(tool="line_bin", coord_list=self.data["sequences"])
                            self._canvas_blend()
                            self._canvas_add_lines()
                            self._canvas_mount()
                            
                        ## external window close
                        elif _config.window_close:
                            self.flags.end = True
                            cv2.destroyAllWindows()

                        
    def _comment_tool(self):
                
        if self.keypress > 0 and not self.keypress in [8, 13, 27]:
            self.data[settings._comment_type] = self.data[settings._comment_type] + chr(self.keypress)
        elif self.keypress == 8:
            self.data[settings._comment_type] = self.data[settings._comment_type][0 : len(self.data[settings._comment_type]) - 1]

        self.canvas = copy.deepcopy(self.canvas_copy)
        cv2.putText(
            self.canvas,
            "Enter " + self.label + ": " + self.data[settings._comment_type],
            (int(self.canvas.shape[0] // 10), int(self.canvas.shape[1] / 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.settings.label_size,
            self.settings.label_colour,
            self.settings.label_width,
            cv2.LINE_AA,
        )
        cv2.imshow(self.settings.window_name, self.canvas)
                    
    def _on_mouse_plain(self, event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEWHEEL and not self.keypress == 9:
            self.keypress = None
            if flags > 0:
                if self.zoom_idx < self.settings.zoom_n_steps:
                    self.flag_zoom = 1
                    self.zoom_idx += 1
                    if self.settings.zoom_mode == "continuous" or (
                        self.settings.zoom_mode == "fixed" and self.zoom_idx == 2
                    ):
                        self._zoom_fun(x, y)
            if flags < 0:
                if self.zoom_idx > 1:
                    self.flag_zoom = -1
                    self.zoom_idx -= 1
                    if self.settings.zoom_mode == "continuous" or (
                        self.settings.zoom_mode == "fixed" and self.zoom_idx == 1
                    ):
                        self._zoom_fun(x, y)
            self.x, self.y = x, y
            cv2.imshow(self.settings.window_name, self.canvas)

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
            self.data[settings._coord_type].append(self.coords_original)
                       
            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(tool="point", coord_list=self.data[settings._coord_type])
            self._canvas_mount()
            
        if event == cv2.EVENT_RBUTTONDOWN:
            
            ## remove points from list, if any are left
            if len(self.data[settings._coord_type]) > 0:
                self.data[settings._coord_type] = self.data[settings._coord_type][:-1]
                
            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(tool="point", coord_list=self.data[settings._coord_type])
            self._canvas_mount()
                

    def _on_mouse_polygon(self, event, x, y, flags, **kwargs):

        ## kwargs
        polyline = kwargs.get("polyline", False)
        reference = kwargs.get("reference", False)
        flag_draw = kwargs.get("draw", False)

        if event == cv2.EVENT_MOUSEMOVE:
            if (reference or flag_draw) and self.tool == "line" and len(self.data[settings._coord_type]) == 2:
                return
            
            ## draw line between current cursor coords and last polygon node
            if len(self.data[settings._coord_type]) > 0:
                self.coords_prev = (
                    int((self.data[settings._coord_type][-1][0] - self.zoom_x1) / self.global_fx),
                    int((self.data[settings._coord_type][-1][1] - self.zoom_y1) // self.global_fy),
                )
                self.canvas = copy.deepcopy(self.canvas_copy)
                cv2.line(
                    self.canvas,
                    self.coords_prev,
                    (x, y),
                    self.settings.line_colour,
                    self.settings.line_width,
                )
                
            ## if in reference mode, don't connect
            elif (reference or flag_draw) and self.tool == "line" and len(self.data[settings._coord_type]) > 2:
                pass
            
            ## pump updates
            cv2.imshow(self.settings.window_name, self.canvas)
            
        if event == cv2.EVENT_LBUTTONDOWN:  
        
            ## skip if in reference mode
            if reference and len(self.data[settings._coord_type]) == 2:
                print("already two points selected")
                return
            
            ## convert cursor coords from zoomed canvas to original coordinate space
            self._zoom_coords_orig(x,y)
            
            ## append points to point list
            self.data[settings._coord_type].append(self.coords_original)
            
            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="line", coord_list=self.data["polygons"] + [self.data[settings._coord_type]])
            self._canvas_mount()
            
            ## if in reference mode, append to ref coords
            if reference and len(self.data[settings._coord_type]) == 2:
                print("Reference set")
                self.data["reference_coords"] = self.data[settings._coord_type]
                                                
        if event == cv2.EVENT_RBUTTONDOWN:
            
            ## remove points and update canvas
            if len(self.data[settings._coord_type]) > 0:
                self.data[settings._coord_type] = self.data[settings._coord_type][:-1]
            else:
                self.data["polygons"] = self.data["polygons"][:-1]
                
            ## apply tool and refresh canvas
            print("remove")
            self._canvas_renew()
            self._canvas_draw(
                tool="line", coord_list=self.data["polygons"] + [self.data[settings._coord_type]])
            self._canvas_mount()

        if flags == cv2.EVENT_FLAG_CTRLKEY and len(self.data[settings._coord_type]) > 2:

            ## close polygon
            if not polyline:
                self.data[settings._coord_type].append(self.data[settings._coord_type][0])
                
            ## add current points to polygon and empyt point list
            print("poly")
            self.data["polygons"].append(self.data[settings._coord_type])
            self.data[settings._coord_type] = []

            ## apply tool and refresh canvas
            self._canvas_renew()
            self._canvas_draw(
                tool="line", coord_list=self.data["polygons"] + [self.data[settings._coord_type]])
            self._canvas_mount()


    def _on_mouse_rectangle(self, event, x, y, flags, **kwargs):
        
        ## kwargs
        template = kwargs.get("template", False)
        
        if event == cv2.EVENT_LBUTTONDOWN:  
            
            ## end after one set of points if creating a template
            if template == True and len(self.data["polygons"]) == 1:
                return
            
            ## start drawing temporary rectangle 
            self.flags.rect_start = x, y
            self.canvas_copy = copy.deepcopy(self.canvas)
            
        if event == cv2.EVENT_LBUTTONUP:
            
            ## end after one set of points if creating a template
            if template == True and len(self.data["polygons"]) == 1:
                print("Template selected")
                return
            
            ## end drawing temporary rectangle
            self.flags.rect_start = None

            ## convert rectangle to polygon coords
            self.rect = [
                int(self.zoom_x1 + (self.global_fx * self.rect_minpos[0])),
                int(self.zoom_y1 + (self.global_fy * self.rect_minpos[1])),
                int(self.zoom_x1 + (self.global_fx * self.rect_maxpos[0])),
                int(self.zoom_y1 + (self.global_fy * self.rect_maxpos[1])),
                ]
            self.data["polygons"].append(
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
                tool="line", coord_list=self.data["polygons"])
            self._canvas_mount(refresh=False)
            
        if event == cv2.EVENT_RBUTTONDOWN:
            
            ## remove polygons and update canvas
            if len(self.data["polygons"]) > 0:
                self.data["polygons"] = self.data["polygons"][:-1]
                
                ## apply tool and refresh canvas
                self._canvas_renew()
                self._canvas_draw(
                    tool="line", coord_list=self.data["polygons"])
                self._canvas_mount()

                
        ## draw temporary rectangle
        elif self.flags.rect_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:        
                self.canvas = copy.deepcopy(self.canvas_copy)
                self.rect_minpos = min(self.flags.rect_start[0], x), min(self.flags.rect_start[1], y)
                self.rect_maxpos = max(self.flags.rect_start[0], x), max(self.flags.rect_start[1], y)
                cv2.rectangle(
                    self.canvas,
                    self.rect_minpos,
                    self.rect_maxpos,
                    self.settings.line_colour,
                    self.settings.line_width,
                )
                cv2.imshow(self.settings.window_name, self.canvas)
                
                
    def _on_mouse_draw(self, event, x, y, flags):     

        ## set colour - left/right mouse button use different settings.colours
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.colour_current_bin = 255
                self.colour_current = self.settings.overlay_colour_left
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.colour_current_bin = 0
                self.colour_current = self.settings.overlay_colour_right
            
            ## start drawing and use current coords as start point
            self.canvas = copy.deepcopy(self.canvas_copy)
            
            ## convert cursor coords from zoomed canvas to original coordinate space
            self.ix,self.iy=x,y
            self.coords_original_i = (
                int(self.zoom_x1 + (self.ix * self.global_fx)),
                int(self.zoom_y1 + (self.iy * self.global_fy)),
            )
            self.data[settings._coord_type].append(self.coords_original_i)
            self.flags.drawing=True

        ## finish drawing and update image_copy
        if event==cv2.EVENT_LBUTTONUP or event==cv2.EVENT_RBUTTONUP:
            self.flags.drawing=False
            self.canvas = copy.deepcopy(self.canvas_copy)
            self.data["sequences"].append([
                self.data[settings._coord_type],
                self.colour_current_bin, 
                int(self.settings.line_width*self.global_fx),
                ])
            self.data[settings._coord_type] = []
            
            ## draw all segments
            self._canvas_renew()
            self._canvas_draw(tool="line_bin", coord_list=self.data["sequences"])
            self._canvas_blend()
            self._canvas_add_lines()
            self._canvas_mount()
                
        ## drawing mode
        elif self.flags.drawing:

            ## convert cursor coords from zoomed canvas to original coordinate space
            self._zoom_coords_orig(x,y)
            
            ## add points, colour, and line width to point list
            self.data[settings._coord_type].append(self.coords_original)
            
            ## draw onto canvas for immediate feedback
            cv2.line(self.canvas,(self.ix,self.iy),(x,y), 
                     self.colour_current, self.settings.line_width) 
            self.ix,self.iy = x,y
            cv2.imshow(self.settings.window_name, self.canvas)  
                        
        if self.keypress == 9 and event == cv2.EVENT_MOUSEWHEEL:
            if flags > 1:
                self.line_width_orig += 1
            if flags < 1 and self.line_width_orig > 1:
                self.line_width_orig -= 1

            self.canvas = copy.deepcopy(self.canvas_copy)
            self.settings.line_width = int(
                self.line_width_orig / ((self.zoom_x2 - self.zoom_x1) / self.image_width))
            cv2.line(self.canvas, (x, y), (x, y),
                     _get_bgr("black"), self.settings.line_width)
            cv2.line(self.canvas, (x, y), (x, y),
                     _get_bgr("white"), max(self.settings.line_width-5, 1))
            cv2.imshow(self.settings.window_name, self.canvas)
            
    
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
                thickness=self.settings.overlay_line_width,
                color=self.settings.overlay_colour_left,
                maxLevel=3,
                offset=None,
            )   
            
            
    def _canvas_blend(self):
        
        ## create coloured overlay from binary image
        self.colour_mask = copy.deepcopy(self.image_bin_copy)
        self.colour_mask = cv2.cvtColor(self.colour_mask, cv2.COLOR_GRAY2BGR)
        self.colour_mask[self.image_bin_copy == 0] = self.settings.overlay_colour_right
        self.colour_mask[self.image_bin_copy == 255] = self.settings.overlay_colour_left

        ## blend two canvas layers
        self.image_copy = cv2.addWeighted(self.image_copy,
                                          1 - self.settings.overlay_blend,
                                          self.colour_mask,
                                          self.settings.overlay_blend,
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
                    self.settings.line_colour,
                    self.settings.line_width,
                )
            elif tool == "line_bin":
                cv2.polylines(
                    self.image_bin_copy,
                    np.array([coords[0]]),
                    False,
                    coords[1],
                    coords[2],
                )
            elif tool == "point":
                cv2.circle(
                    self.image_copy,
                    coords,
                    self.settings.point_size,
                    self.settings.point_colour,
                    -1,
                )
                if self.label:                    
                    cv2.putText(
                        self.image_copy,
                        str(idx+1),
                        coords,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.settings.label_size,
                        self.settings.label_colour,
                        self.settings.label_width,
                        cv2.LINE_AA,
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
        if refresh and not self.passive:
            cv2.imshow(self.settings.window_name, self.canvas)


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
            self.settings.line_width = int(self.line_width_orig / ((self.zoom_x2 - self.zoom_x1) / self.image_width))


    def _zoom_coords_orig(self, x, y):
        self.coords_original = (
                int(self.zoom_x1 + (x * self.global_fx)),
                int(self.zoom_y1 + (y * self.global_fy)),
            )

        
class _NoIndent(object):
    
    def __init__(self, value):
        # if not isinstance(value, (list, tuple, dict)):
        #     raise TypeError('Only lists and tuples can be wrapped')
        self.value = value


class _NoIndentEncoder(json.JSONEncoder):
    
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(_NoIndentEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, _NoIndent)
                    else super(_NoIndentEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        
        if isinstance(obj, np.intc):
            return int(obj)
        
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(_NoIndentEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded
        

class _YamlFileMonitor:
    def __init__(self, filepath, delay=500):

        filepath = os.path.abspath(filepath)        

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
        
        
#%% functions - ANNOTATION helpers

def _get_annotation(
        annotations, 
        annotation_type,
        annotation_id=None,
        reduce_counter=False,
        prep_msg=None,
        kwargs={},
        ):
    
    ## setup    
    pype_mode = kwargs.get("pype_mode", False)
    prep_msg = kwargs.get("prep_msg", "")
    
    annotations = copy.deepcopy(annotations)
    
    if not annotation_type.__class__.__name__ == "NoneType":
        annotation_id_str = annotation_type + "_id"
        print_msg = ""
    else:
        return {}
    
    ## get non-generic id for plotting
    if annotation_id_str in kwargs:
        annotation_id = kwargs.get(annotation_id_str)
        
    if annotations.__class__.__name__ in ["dict", "defaultdict"]:
                
        ## get ID from last used annotation function of that type
        if annotation_id.__class__.__name__ == "NoneType":
            
            
            if kwargs.get("annotation_counter"):
                print_msg = "- \"{}\" not provided: ".format(annotation_id_str)
                annotation_counter = kwargs.get("annotation_counter")
                annotation_id = string.ascii_lowercase[annotation_counter[annotation_type]]
                if annotation_id == "z":
                    print_msg = print_msg + "- no precursing annotations of type \"{}\" found".format(annotation_type)
                    annotation_id = None
                else:
                    if reduce_counter:
                        annotation_id =  chr(ord(annotation_id) - 1)
                    print_msg = print_msg + "using last annotation of type \"{}\" with ID \"{}\"".format(annotation_type, annotation_id)
            else:
                if annotation_type in annotations:
                    annotation_id = max(list(annotations[annotation_type].keys()))
                    print_msg = "\"{}\" not specified - using endmost in provided annotations: \"{}\"".format(annotation_id_str, annotation_id)

                else:
                    annotation = {}
                    # print_msg = "\"{}\" not specified and annotation type not found".format(annotation_id_str)

        ## check if type is given
        if annotation_type in annotations:
                                            
            ## extract item
            if annotation_id:
                if annotation_id in annotations[annotation_type]:
                    annotation = annotations[annotation_type][annotation_id]
                else:
                    print_msg = "could not find \"{}\" with ID \"{}\"".format(annotation_type, annotation_id)
                    annotation = {}
            else:
                print("NONE")
                annotation = {}
        else:
            # print_msg = "incompatible annotation type supplied - need \"{}\" type".format(annotation_type)
            annotation = {}
            
        ## cleaned feedback (skip identical messages)
        while True:
            if print_msg:
                if prep_msg:
                    print_msg = prep_msg + "\n\t" + print_msg
                if pype_mode:          
                    if not print_msg == _config.last_print_msg:
                        _config.last_print_msg = print_msg
                        break
                    else:
                        pass
                else:
                    pass
                print(print_msg)
                break
            break
    else:
        annotation = {}
                
    return annotation
              		

def _get_annotation_type(fun_name):
    
    return settings._annotation_functions[fun_name]


def _get_GUI_data(annotation):
    
    data = []
            
    if annotation:
        if "info" in annotation:
            annotation_type = annotation["info"]["annotation_type"]
        if "data" in annotation:
            data = annotation["data"][annotation_type]
    
    
    return data


def _get_GUI_settings(kwargs, annotation=None):
    
    GUI_settings = {}
    
    if annotation:
        if "settings" in annotation:
            if "GUI" in annotation["settings"]:
                for key, value in annotation["settings"]["GUI"].items():
                    GUI_settings[key] = value
        
    if kwargs:
        for key, value in kwargs.items():
            if key in settings._GUI_settings_args:
                GUI_settings[key] = value
            elif key in ["passive"]:
                GUI_settings[key] = value
    
    return GUI_settings


def _update_annotations(
        annotations, 
        annotation,
        annotation_type,
        annotation_id,
        kwargs,
        ):
                
    annotations = copy.deepcopy(annotations)
    
    if not annotation_type in annotations:
        annotations[annotation_type] = {}
        
    if annotation_id.__class__.__name__ == "NoneType": 
        if "annotation_counter" in kwargs:
            annotation_counter = kwargs.get("annotation_counter")
            annotation_id = string.ascii_lowercase[annotation_counter[annotation_type]]
        else:
            annotation_id = "a"
            
    annotations[annotation_type][annotation_id] = copy.deepcopy(annotation)
                    
    return annotations



#%% functions - GUI helpers


def _auto_line_width(image, **kwargs):
    factor = kwargs.get("factor", settings.auto_line_width_factor)
    image_height, image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) / 2
    line_width = max(int(factor * image_diagonal), 1)

    return line_width


def _auto_point_size(image, **kwargs):
    factor = kwargs.get("factor", settings.auto_point_size_factor)
    image_height, image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) / 2
    point_size = max(int(factor * image_diagonal), 1)

    return point_size


def _auto_text_width(image, **kwargs):
    factor = kwargs.get("factor", settings.auto_text_width_factor)
    image_height, image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) / 2
    text_width = max(int(factor * image_diagonal), 1)

    return text_width


def _auto_text_size(image, **kwargs):
    factor = kwargs.get("factor", settings.auto_text_size_factor)
    image_height, image_width = image.shape[0:2]
    image_diagonal = (image_height + image_width) / 2
    text_size = max(int(factor * image_diagonal), 1)

    return text_size


def _get_bgr(col_string):
    col = Color(col_string)
    rgb = col.get_rgb()
    rgb_255 = []
    for component in rgb:
        rgb_255.append(int(component * 255))
        
    return tuple((rgb_255[2], rgb_255[1], rgb_255[0]))


#%% functions - YAML helpers

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
    yaml.indent(mapping=4, sequence=4, offset=4)

    if isinstance(filepath, (Path, str)):
        if Path(filepath).is_file():
            with open(filepath, "r") as file:
                return yaml.load(file)
            
        else:
            print("Cannot load config from specified filepath")
    else:
        print("Not a valid path - couldn't load yaml.")
        return
    
    
def _show_yaml(odict, ret=False, typ="rt"):
    
    yaml =  YAML(typ=typ)
    yaml.indent(mapping=4, sequence=4, offset=4)

    if ret:
        with io.StringIO() as buf, redirect_stdout(buf):
            yaml.dump(odict, sys.stdout)
            return buf.getvalue()
    else:
        yaml.dump(odict, sys.stdout)
    


def _save_yaml(dictionary, filepath, typ="rt"):
    yaml = YAML(typ=typ)
    yaml.indent(mapping=4, sequence=4, offset=4)
    with open(filepath, "w") as out:
        yaml.dump(dictionary, out)


def _yaml_flow_style(dictionary):
   ret = ruamel.yaml.comments.CommentedMap(dictionary)
   ret.fa.set_flow_style()
   return ret   


def _yaml_recursive_delete_comments(d):
    if isinstance(d, dict):
        for k, v in d.items():
            _yaml_recursive_delete_comments(k)
            _yaml_recursive_delete_comments(v)
    elif isinstance(d, list):
        for elem in d:
            _yaml_recursive_delete_comments(elem)
    try:
         # literal scalarstring might have comment associated with them
         attr = 'comment' if isinstance(d, ruamel.yaml.scalarstring.ScalarString) \
                  else ruamel.yaml.comments.Comment.attrib 
         delattr(d, attr)
    except AttributeError:
        pass


#%% functions - VARIOUS





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



def _check_pype_tag(tag):
    
    if tag.__class__.__name__ == "str":
        
        ## pype name check
        if "pype_config" in tag:
            tag = tag.replace("pype_config", "")
            print("Do not add \"pype_config\", only a short tag")
        if ".yaml" in tag:
            tag = tag.replace(".yaml", "")
            print("Do not add extension, only a short tag")
        if "_" in tag:
            raise SyntaxError("Underscore not allowed in pype tag - aborting.")
        for char in "[@!#$%^&*()<>?/|}{~:]\\":
            if char in tag:
                raise SyntaxError("No special characters allowed in pype tag - aborting.")
    

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
    if contours[0].__class__.__name__ == "list" or contours.__class__.__name__ == "list":
        cv2.fillPoly(mask_bin, [np.array(contours, dtype=np.int32)], _get_bgr("white"))
    elif contours[0].__class__.__name__ == "ndarray":
        for contour in contours:
            cv2.fillPoly(mask_bin, [np.array(contour, dtype=np.int32)], _get_bgr("white"))
    return mask_bin


def _create_mask_bool(image, contours):
    mask_bin = _create_mask_bin(image, contours)
    return np.array(mask_bin, dtype=bool)


def _decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def _del_rw(action, name, exc):
    os.chmod(name, S_IWRITE)
    os.remove(name)
          
    
    
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
                if pype_mode:
                    if any(inc in Path(filepath).stem for inc in include):
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








def _load_project_image_directory(dir_path, tag=None, as_container=True, **kwargs):
    """
    Parameters
    ----------
    dirpath: str
        path to a phenopype project directory containing raw image, attributes 
        file, masks files, results df, etc.
    tag : str
        pype suffix that is appended to all output
        
    Returns
    -------
    container
        A phenopype container is a Python class where loaded images, 
        dataframes, detected contours, intermediate output, etc. are stored 
        so that they are available for inspection or storage at the end of 
        the analysis. 

    """
    
    ## check if directory
    if not os.path.isdir(dir_path):
        print("Not a valid phenoype directory - cannot load files.")
        return
    
    ## check if attributes file and load otherwise
    if not os.path.isfile(os.path.join(dir_path, "attributes.yaml")):
        print("Attributes file missing - cannot load files.")
        return
    else:
        attributes = _load_yaml(os.path.join(dir_path, "attributes.yaml"))
    
    ## check if requires info is contained in attributes and load image
    if not "image_phenopype" in attributes or not "image_original" in attributes:
        print("Attributes doesn't contain required meta-data - cannot load files.")
        return 

    ## load image
    if attributes["image_phenopype"]["mode"] == "link":
        image_path = attributes["image_original"]["filepath"]
    else:
        image_path = os.path.join(dir_path,attributes["image_phenopype"]["filename"])
    image = utils.load_image(image_path)

    ## return
    if as_container:
        return utils.Container(image=image, dir_path=dir_path, file_suffix=tag)
    else:
        return image



def _load_image_data(image_path, path_and_type=True, resize=1):
    """
    Create a DataFreame with image information (e.g. dimensions).

    Parameters
    ----------
    image: str or ndarray
        can be a path to an image stored on the harddrive OR an array already 
        loaded to Python.
    path_and_type: bool, optional
        return image path and filetype to image_data dictionary

    Returns
    -------
    image_data: dict
        contains image data (+meta data, if selected)

    """
    
    
    if image_path.__class__.__name__ == "str":
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            width, height = image.size
            image.close()
            image_data = {
                "filename": os.path.split(image_path)[1],
                "width": width,
                "height": height,
            }
            
            if path_and_type: 
                image_data.update({
                    "filepath": image_path,
                    "filetype": os.path.splitext(image_path)[1]})
        else:
            raise FileNotFoundError("Invalid image path - could not load image.")
    else:
        raise TypeError("Not a valid image file - cannot read image data.")

    # Image.MAX_IMAGE_PIXELS = 999999999

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





def _resize_image(image, factor=1, interpolation="cubic"):
    """
    Resize image by resize factor 

    Parameters
    ----------
    image: array 
        image to be resized
    resize: float, optional
        resize factor for the image (1 = 100%, 0.5 = 50%, 0.1 = 10% of 
        original size).
    interpolation: str, optional
        interpolation algorithm to use. check pp.settings.settings.opencv_interpolation_flags
        and refer to https://docs.opencv.org/3.4.9/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

    Returns
    -------
    image : ndarray
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


def _save_prompt(object_type, filepath, ow_flag):
    
    if os.path.isfile(filepath) and ow_flag == False:
        print_msg = "- {} not saved - file already exists (overwrite=False)".format(object_type)
        ret = False
    elif os.path.isfile(filepath) and ow_flag == True:
        print_msg =  "- {} saved under {} (overwritten)".format(object_type, filepath)
        ret = True
    elif not os.path.isfile(filepath):
        print_msg =  "- {} saved under {}".format(object_type, filepath)
        ret = True

    print(print_msg)
    return ret
    

#%% obsolete?


class _DummyClass:
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)
        

def _update_settings(kwargs, local_settings, IV_settings=None):

    for key, value in kwargs.items():
        if key in settings._image_viewer_arg_list:
            if not IV_settings.__class__.__name__ == "NoneType":
                IV_settings[key] = value
            local_settings[key] = value
    if "passive" in local_settings:
        del local_settings["passive"]



def _drop_dict_entries(dictionary, drop=[]):
    
    new_dictionary = {}
    
    for key, value in dictionary.items():
        if not key in drop:
            new_dictionary[key] = value
        
    return new_dictionary


def _load_previous_annotation(annotation_previous, components, load_settings=True):
    ImageViewer_previous = {}    
    if load_settings:
        ImageViewer_previous.update(annotation_previous["settings"])
    for item in components:
        field, data = item
        ImageViewer_previous[data] = annotation_previous[field][data]
        
    return _DummyClass(ImageViewer_previous)


def _calc_circle_perimeter(center_x, center_y, radius):
    coordinate_list=[]
    for i in range(360):
        y = center_x + radius * cos(i)
        x = center_y + radius * cos(i)
        coordinate_list.append((int(x),int(y)))
        
    return coordinate_list


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
