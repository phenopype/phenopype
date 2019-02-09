from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import numpy as np
import numpy.ma as ma
import pandas as pd
import cv2
import os
import pprint
import copy

from phenopype.utils import (blur, decode_fourcc)
from phenopype.utils import (red)

#%% classes
             
       
class motion_tracker(object):
    def __init__(self, video_path, **kwargs): 
        """Read properties of input video file.
        """
        if isinstance(video_path, str):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()

        else:
            print("Error - full or relative path of video-file needed")
        
        self.frame = frame
        self.path = video_path       
        self.name = os.path.basename(self.path)
        self.nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.fourcc_str = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        self.length = str(int(( self.nframes / self.fps)/60)).zfill(2) + ":" +str(int((((self.nframes / self.fps)/60)-int((self.nframes / self.fps)/60))*60)).zfill(2)
        self.dimensions = tuple(reversed(frame.shape[0:2]))
        if frame.shape[2] == 3:
            self.is_colour = True
            
        cap.release()
        
        print("\n")
        print("----------------------------------------------------------------")
        print("Input video properties - \"" + self.name + "\":\n")
        print("Frames per second: " + str(self.fps) + "\nN frames: " + str(self.nframes) + "\nLength: " + str(self.length) + " (mm:ss)" + "\nDimensions: " + str(self.dimensions) + "\nColour video: " + str(self.is_colour) + "\nFourCC code: " + str(self.fourcc_str)) 
        print("----------------------------------------------------------------")


    def video_output(self, **kwargs):
        """Set properties of output video file .
        """
        name_out = kwargs.get("name",  os.path.splitext(self.name)[0] + "_" + kwargs.get("suffix", "out") + ".avi")  
        dir_out = kwargs.get("save_dir", os.path.dirname(self.path))
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        path_out = os.path.join(dir_out, name_out)
            
        fps_out = kwargs.get("fps", self.fps)
        dimensions_out = kwargs.get("dimensions", self.dimensions)
        self.resize_factor = kwargs.get("resize", 1)
        dimensions_out = (int(dimensions_out[0] * self.resize_factor), int(dimensions_out[1] * self.resize_factor))
        if self.resize_factor < 1:
            res_msg = " (original resized by factor " + str(self.resize_factor) + ")"
        else:
            res_msg = ""
        fourcc_str_out = kwargs.get("fourcc_string", "DIVX")
        fourcc_out = cv2.VideoWriter_fourcc(*fourcc_str_out)
        save_colour = kwargs.get("save_colour", self.is_colour)
        
        self.save_video = kwargs.get("save_video","True")
        self.writer = cv2.VideoWriter(path_out, fourcc_out, fps_out, dimensions_out, save_colour) 


        print("\n")
        print("----------------------------------------------------------------")
        print("Output video settings - \"" + self.name + "\":\n")
        print("Save name: " + name_out + "\nSave dir: " + dir_out + "\nFrames per second: " + str(fps_out) + "\nDimensions: " + str(dimensions_out) + res_msg +  "\nColour video: " + str(save_colour) + "\nFormat (FourCC code): " + fourcc_str_out) 
        print("----------------------------------------------------------------")
        

    def motion_detection(self, **kwargs):

        self.skip = kwargs.get("skip", 5)
        self.wait = kwargs.get("start_after", 15)
        self.finish = kwargs.get("finish_after", 0)

        history = kwargs.get("history", 60)
        backgr_thresh = kwargs.get("backgr_thresh", 10)        
        self.detect_shadows = kwargs.get("detect_shadows", True)

        self.fgbg_subtractor = cv2.createBackgroundSubtractorMOG2(history = int(history * (self.fps / self.skip)), varThreshold = int(backgr_thresh), detectShadows = self.detect_shadows)
        
        if "methods" in kwargs:
            self.methods = kwargs.get("methods")
            for m in self.methods:
                m.print_settings()
                

        include = kwargs.get("mask_include", [])
        self.layers = include
        self.layer_names = kwargs.get("mask_names", [])

        exclude = kwargs.get("mask_exclude", [])
               
        
        in_mask = np.zeros(self.frame.shape[0:2], dtype=bool)
        for i in include:
            in_mask = np.logical_or(in_mask, i)
            
        ex_mask = np.zeros(self.frame.shape[0:2], dtype=bool)
        for i in exclude:
            ex_mask = np.logical_or(ex_mask, i)
        
        self.mask = np.add(in_mask.astype(np.uint8), ex_mask.astype(np.uint8))
        self.mask[self.mask==2]=0
        self.mask[self.mask==1]=255
        
        print("\n")
        print("----------------------------------------------------------------")
        print("Motion detection settings - \"" + self.name + "\":\n")
        print("\n\"History\"-parameter: " + str(history) + " seconds" + "\nSensitivity: " + str(backgr_thresh) + "\nRead every nth frame: " + str(self.skip) + "\nDetect shadows: " + str(self.detect_shadows) + "\nStart after n seconds: " + str(self.wait) + "\nFinish after n seconds: " + str(self.finish if self.finish > 0 else " - ")) 
        print("----------------------------------------------------------------")
              

        
    def run_tracking(self, **kwargs):
        
        weight = kwargs.get("weight", 0.5)
        show_selector = kwargs.get("show","overlay")
        return_df = kwargs.get("return_df","False")

        self.df = pd.DataFrame()
        self.idx1, self.idx2 = (0,0)
        self.capture = cv2.VideoCapture(self.path)        
        self.wait_frame = self.wait * self.fps
        if self.finish > 0:
            self.finish_frame = self.finish * self.fps
            

        #methods_out = []
        while(self.capture.isOpened()):
            
            # read video 
            ret, self.frame = self.capture.read()     
            self.capture_frame = False
            if ret==False:
                break
            
            # indexing 
            self.idx1, self.idx2 = (self.idx1 + 1,  self.idx2 + 1)    
            if self.idx2 == self.skip:
                self.idx2 = 0    
            if self.finish > 0:
                if self.idx1 >= self.finish_frame:
                    break
                             
                
            # warmup fgbg-algorithm shortly before capturing 
            if self.idx1 > self.wait_frame - (3*self.fps) and self.idx2 == 0:
                self.fgmask = self.fgbg_subtractor.apply(self.frame)               
                
            # start modules after x seconds
            if self.idx1 > self.wait_frame and self.idx2 == 0:
                self.capture_frame = True
                           
            # feedback
            mins = str(int((self.idx1 / self.fps)/60)).zfill(2)
            secs = str(int((((self.idx1 / self.fps)/60)-int(mins))*60)).zfill(2)    
            self.time_stamp = "Time: " + mins + ":" + secs + "/" + self.length + " - Frames: " + str(self.idx1) + "/" + str(int(self.nframes))
            
            
            # CAPTURE FRAME
            if self.capture_frame == True:    

                self.frame_overlay = self.frame    
                
                if len(self.layers) > 0:
                    self.fgmask = np.bitwise_and(self.mask, self.fgmask)
                else:
                    self.layers = []
                    self.layer_names = []
                    
                if "methods" in vars(self):
                    for m in self.methods:
                        self.overlay, self.fgmask_processed, self.frame_df = m._run(frame=self.frame, fgmask=self.fgmask, layers=self.layers, layer_names=self.layer_names)
                        self.frame_overlay = cv2.addWeighted(self.frame_overlay, 1, self.overlay, weight, 0)    
                        
                        # DATA FRAME
                        self.frame_df.insert(0, 'frame_abs',  self.idx1)
                        self.frame_df.insert(1, 'frame_rel',  int((self.idx1-self.wait_frame)/self.skip))
                        self.frame_df.insert(2, 'mins',  mins)
                        self.frame_df.insert(3, 'secs',  secs)
                        self.df = self.df.append(self.frame_df, ignore_index=True, sort=False)                     
                
                # SHOW OUTPUT
                if show_selector == "overlay":
                    self.show = self.frame_overlay
                elif show_selector == "mask":
                    self.show = self.fgmask    
                elif show_selector == "mask_processed":
                    self.show = self.fgmask_processed
                else:
                    self.show = self.frame
                self.show = cv2.resize(self.show, (0,0), fx=self.resize_factor, fy=self.resize_factor) 
                
                cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
                cv2.imshow('phenopype', self.show)   
                                
                # SAVE OUTPUT
                if self.save_video == True:
                    self.writer.write(self.show)
                
     
                print(self.time_stamp + " - captured")                
            else:
                print(self.time_stamp)

            # keep stream open
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # cleanup
        self.capture.release()
        self.writer.release()
        cv2.destroyAllWindows()
        
        if return_df == True:
            return self.df
        
        
class tracking_method(object):
    def __init__(self, **kwargs):
        """Constructs a tracking method that can be supplied to the tracker. 
        """                   
        for key, value in kwargs.items():
            if key in kwargs:
                setattr(self, key, value)
                
        if not "min_length" in vars(self):
            self.min_length = 1
        if not "max_length" in vars(self):
            self.max_length = 1000           
        if not "overlay_colour" in vars(self):
            self.overlay_colour = red
        if not "mode" in vars(self):
            self.mode = "multiple"
        if not "operations" in vars(self):
            self.operations = []
            
    def print_settings(self, **kwargs):
        """Prints the settings of the tracking method. 
        """ 
        width = kwargs.get("width",30)
        indent = kwargs.get("indent",1)
        compact = kwargs.get("compact",True)
        pretty = pprint.PrettyPrinter(width=width, compact=compact, indent=indent)
        
        pretty.pprint(vars(self))
        
            
    def _run(self, frame, fgmask, **kwargs):
        """Run tracking method on current frame. Internal reference - don't call this directly.
        """
        self.frame = frame
        self.fgmask = fgmask
        overlay = np.zeros_like(self.frame) 
        layers = kwargs.get("layers", [])
        layer_names = kwargs.get("layer_names", [])

        if "remove_shadows" in vars(self):
            if self.remove_shadows==True:
                ret, self.fgmask = cv2.threshold(self.fgmask, 128, 255, cv2.THRESH_BINARY)
         
        if "blur" in vars(self):
            blur_kernel, thresh_val = (self.blur)
            self.fgmask = blur(self.fgmask, blur_kernel)            
            ret, self.fgmask = cv2.threshold(self.fgmask, thresh_val, 255, cv2.THRESH_BINARY)  

        self.fgmask_processed = self.fgmask             

        # =============================================================================
        # find objects
        # =============================================================================
        
        ret, contours, hierarchy = cv2.findContours(self.fgmask_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1) #CHAIN_APPROX_NONE
        
        if len(contours) > 0:      
        
            list_contours = []
            list_length = []         
            list_coordinates = [] 
            
            df_list = []
            df_column_names = []
            
            # check if contour matches min/max length provided
            for contour in contours:
                if contour.shape[0] > 4:
                    center,radius = cv2.minEnclosingCircle(contour)
                    length = int(radius * 2)     
                    if length < self.max_length and length > self.min_length:
                        list_length.append(length)      
                        list_contours.append(contour) 
                        list_coordinates.append(center) 

            # if single biggest organism:
            if len(list_contours) > 0:      

                if self.mode == "single":
                    if len(contours)==1:
                        pass
                    elif len(contours)>1:
                        max_idx = np.argmax(list_length)
                        list_contours = [list_contours[max_idx]]
                        list_length = [list_length[max_idx]]
                        list_coordinates = [list_coordinates[max_idx]]
    

    
                list_area, list_x, list_y = [],[],[]
                list_grayscale, list_grayscale_background = [],[]
                list_b, list_g, list_r = [],[],[] 
                list_layers = []
                for contour, coordinate in zip(list_contours, list_coordinates):
                    
                    # operations    
                    x=int(coordinate[0])
                    y=int(coordinate[1])
                    list_x.append(x)
                    list_y.append(y)
                    
                    if len(layers)>0:
                        temp_list = []
                        for i in layers:
                            temp_list.append(i[y,x])                        
                        list_layers.append(temp_list)

                    rx,ry,rw,rh = cv2.boundingRect(contour)
                    frame_roi = self.frame[ry:ry+rh,rx:rx+rw]
                    frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                    mask_roi = self.fgmask_processed[ry:ry+rh,rx:rx+rw]               
                    
                    if any("area" in o for o in self.operations):
                        list_area.append(int(cv2.contourArea(contour)))
            
                    if any("grayscale" in o for o in self.operations):
                        grayscale =  ma.array(data=frame_roi_gray, mask = np.logical_not(mask_roi))
                        list_grayscale.append(int(np.mean(grayscale)))
                        
                    if any("grayscale_background" in o for o in self.operations):
                        grayscale_background =  ma.array(data=frame_roi_gray, mask = mask_roi)
                        if not grayscale_background.mask.all():
                            list_grayscale_background.append(int(np.mean(grayscale_background)))
                        else:
                            list_grayscale_background.append(9999)
                            
                    if any("bgr" in o for o in self.operations):
                        b = ma.array(data=frame_roi[:,:,0], mask = np.logical_not(mask_roi))
                        list_b.append(int(np.mean(b)))
                        g = ma.array(data=frame_roi[:,:,1], mask = np.logical_not(mask_roi))
                        list_g.append(int(np.mean(g)))
                        r = ma.array(data=frame_roi[:,:,2], mask = np.logical_not(mask_roi))
                        list_r.append(int(np.mean(r)))                       
                                                                
                    # drawing 
                    overlay = cv2.drawContours(overlay, [contour], 0, self.overlay_colour, -1) # Draw filled contour in mask     
                    overlay = cv2.putText(overlay, self.label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,  1,self.overlay_colour,1,cv2.LINE_AA)  
                    overlay = cv2.rectangle(overlay,(rx,ry),(int(rx+rw),int(ry+rh)),self.overlay_colour,2)
    
                df_list = df_list + [list_x]  
                df_list = df_list + [list_y]  
                df_column_names = df_column_names + ["x","y"]
                
                  

                if any("diameter" in o for o in self.operations):               
                    df_list = df_list + [list_length] 
                    df_column_names.append("diameter")                    
    
                if any("area" in o for o in self.operations):               
                    df_list = df_list + [list_area] 
                    df_column_names.append("area")                    

                if any("grayscale" in o for o in self.operations):
                    df_list = df_list + [list_grayscale] 
                    df_column_names.append("grayscale")
    
                if any("grayscale_background" in o for o in self.operations):
                    df_list = df_list + [list_grayscale_background] 
                    df_column_names.append("grayscale_background")

                if any("bgr" in o for o in self.operations):
                    df_list = df_list + [list_b]  
                    df_list = df_list + [list_g]  
                    df_list = df_list + [list_r]              
                    df_column_names = df_column_names + ["b", "g", "r"]
                    
                frame_df = pd.DataFrame(data=df_list)
                frame_df = frame_df.transpose()                        
                frame_df.columns = df_column_names
                frame_df["label"] = self.label
                
                if len(layers)>0:
                    layer_df = pd.DataFrame(list_layers,columns=layer_names)
                    frame_df = pd.concat([frame_df.reset_index(drop=True), layer_df], axis=1)
                
                self.frame_df = frame_df
                
                return overlay, self.fgmask, frame_df

            else:
                frame_df = pd.DataFrame()
                return overlay, self.fgmask, frame_df
        
        else:
            frame_df = pd.DataFrame()
            return overlay, self.fgmask, frame_df
        
        
        

