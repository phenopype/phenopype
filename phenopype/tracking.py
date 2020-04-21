from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import numpy as np
import numpy.ma as ma
import pandas as pd
import cv2
import os
import pprint

from phenopype.utils_lowlevel import _decode_fourcc
from phenopype.core.segmentation import blur
from phenopype.settings import colours

#%% classes
             
class motion_tracker(object):
    def __init__(self, video_path, at_frame=1, **kwargs): 
        """
        

        Parameters
        ----------
        video_path : TYPE
            DESCRIPTION.
        at_frame : TYPE, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
    
        if isinstance(video_path, str):
            capture = cv2.VideoCapture(video_path)
            idx = 0
            while(capture.isOpened()):
                idx += 1
                if idx == at_frame:
                    ret, frame = capture.read()
                    break
                else:
                    capture.grab()
                if cv2.waitKey(1) & 0xff == 27:
                    break

                    
        else:
            print("Error - full or relative path of video-file needed")
        
        self.frame = frame
        self.path = video_path       
        self.name = os.path.basename(self.path)
        self.nframes = capture.get(cv2.CAP_PROP_FRAME_COUNT) 
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.fourcc_str = _decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
        self.length = str(int(( self.nframes / self.fps)/60)).zfill(2) + ":" + str(int((((self.nframes / self.fps)/60)-int((self.nframes / self.fps)/60))*60)).zfill(2)
        self.dimensions = tuple(reversed(frame.shape[0:2]))
        if frame.shape[2] == 3:
            self.is_colour = True
            
        capture.release()
        
        print("\n")
        print("----------------------------------------------------------------")
        print("Input video properties - \"" + self.name + "\":\n")
        print("Frames per second: " + str(self.fps) + "\nN frames: " + str(self.nframes) + "\nLength: " + str(self.length) + " (mm:ss)" + "\nDimensions: " + str(self.dimensions) + "\nColour video: " + str(self.is_colour) + "\nFourCC code: " + str(self.fourcc_str)) 
        print("----------------------------------------------------------------")


    def video_output(self, **kwargs):
        """Set properties of output video file. Most settings can be left at their default value.
        
        Parameters
        ----------
        video_format: str (default: "DIVX")
            format of the output video file. needs to be a "fourcc"-string: https://www.fourcc.org/codecs.php
        name: str (defaut: input filename + "_out" + input extension)
            name for the output video file
        save_dir: str (default: directory of input file)
            save directory for the output video file. will be created if not existing
        fps: int (default: input fps)
            frames per second of the output video file
        dimensions: tuple (default: input dimensions)
            dimensions (width, length) of the output video file
        resize: int (default: 1 [no resize])
            factor by which to resize the output video dimensions
        save_colour: bool (default: reads input file)
            should the saved video frames be in colour 
        save_video: bool (default: True)
            should an output video file be saved
        """
        ## kward
        fourcc_str_out = kwargs.get("video_format", "DIVX")
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

        self.save_colour = kwargs.get("save_colour", self.is_colour)
        self.save_video = kwargs.get("save_video",True)
        
        fourcc_out = cv2.VideoWriter_fourcc(*fourcc_str_out)
        self.writer = cv2.VideoWriter(path_out, fourcc_out, fps_out, dimensions_out, self.save_colour) 


        print("\n")
        print("----------------------------------------------------------------")
        print("Output video settings - \"" + self.name + "\":\n")
        print("Save name: " + name_out + 
              "\nSave dir: " + dir_out + 
              "\nFrames per second: " + str(fps_out) + 
              "\nDimensions: " + str(dimensions_out) + res_msg +  
              "\nColour video: " + str(self.save_colour) + 
              "\nFormat (FourCC code): " + fourcc_str_out) 
        print("----------------------------------------------------------------")
        

    def motion_detection(self, **kwargs):
        """Set properties of output video file. Most settings can be left at their default value.
        
        Parameters
        ----------
        skip: int (default: 5)
            how many frames to skip between each capture
        start: int (default: 0)
            start after X seconds
        finish: int (default: "video-length")
            finish after X seconds
        history: int (default: 60)
            how many frames to use for fg-bg subtraction algorithm
        backgr_thresh: int (default: 10)
            sensitivity-level for fg-bg subtraction algorithm (lower = more sensitive)
        mode: str (default: "MOG")
            type of fg-bg subtraction algorithm ("MOG" or "KNN")
        methods: list
            list with tracking_method objects
        """
        ## kwargs
        self.skip = kwargs.get("skip", 5)
        self.warmup = kwargs.get("warmup", 0) # currently unsure what this does and why it needed (related to fgbg-detector warmup / quality control)
        self.start = kwargs.get("start_after", 0)
        self.finish = kwargs.get("finish_after", self.nframes)

        history = kwargs.get("history", 60)
        backgr_thresh = kwargs.get("backgr_thresh", 10)        
        self.detect_shadows = kwargs.get("detect_shadows", True)

        mode = kwargs.get("mode", "MOG")
        if mode == "MOG":
            self.fgbg_subtractor = cv2.createBackgroundSubtractorMOG2( int(history * (self.fps / self.skip)),  int(backgr_thresh),  self.detect_shadows)
        elif mode == "KNN":
            self.fgbg_subtractor = cv2.createBackgroundSubtractorKNN( int(history * (self.fps / self.skip)),  int(backgr_thresh),  self.detect_shadows)
        
        if "methods" in kwargs:
            self.methods = kwargs.get("methods")
            for m in self.methods:
                m._print_settings()
                
#        ## currently unsure how this works exactly - keeps masks from masking each other, order matters...        
#        if "consecutive_masking" in kwargs: 
#            self.consecutive = kwargs.get("consecutive_masking")                 
        
        print("\n")
        print("----------------------------------------------------------------")
        print("Motion detection settings - \"" + self.name + "\":\n")
        print("\n\"History\"-parameter: " + str(history) + " seconds" + "\nSensitivity: " + str(backgr_thresh) + "\nRead every nth frame: " +  \
              str(self.skip) + "\nDetect shadows: " + str(self.detect_shadows) + "\nStart after n seconds: " + str(self.start) + \
              "\nFinish after n seconds: " + str(self.finish if self.finish > 0 else " - ")) 
        print("----------------------------------------------------------------")
              

        
    def run_tracking(self, **kwargs):
        """Start motion tracking procedure.
        
        Parameters
        ----------
        show: str (default: overlay)
            show output of tracking as "overlay", binary forground mask "fgmask", or frame by frame
        weight: float (default: 0.5)
            how transparent the overlay should be
        return_df: bool (default: True)
            should the output dataframe be returned or inherited to motion_tracker object
        """      
        weight = kwargs.get("weight", 0.5)
        if "show" in kwargs:
            show_selector = kwargs.get("show")
            show = True
        else:
            show_selector = "overlay"
            show = False
            
        return_df = kwargs.get("return_df","True")
                   
        self.df = pd.DataFrame()
        self.idx1, self.idx2 = (0,0)
        self.capture = cv2.VideoCapture(self.path)        
        self.start_frame = int(self.start * self.fps)
        if self.finish > 0:
            self.finish_frame = int(self.finish * self.fps)
            
        #methods_out = []
        while(self.capture.isOpened()):
            
            # =============================================================================
            # INDEXING AND CONTROL
            # =============================================================================
            
            self.idx1, self.idx2 = (self.idx1 + 1,  self.idx2 + 1)    
            if self.idx2 == self.skip:
                self.idx2 = 0 
                
            mins = str(int((self.idx1 / self.fps)/60)).zfill(2)
            secs = str(int((((self.idx1 / self.fps)/60)-int(mins))*60)).zfill(2)    
            self.time_stamp = "Time: " + mins + ":" + secs + "/" + self.length + " - Frames: " + str(self.idx1) + "/" + str(int(self.nframes))  
        

            # end-control              
            if self.idx1 == self.nframes-1: 
                self.capture.release()
                self.writer.release()
                break            
            if self.idx1 == self.finish_frame-1:
                self.capture.release()
                self.writer.release()
                break                             
          
            # check if frame is empty and if it should be captured
            if self.idx1 > self.start_frame - int(self.warmup * self.fps) and self.idx2 == 0:
                self.ret, self.frame = self.capture.read()  
                if self.ret==False: # skip empty frames
                    continue  
                else:
                    capture_frame = True
                    if self.idx1 < self.start_frame and self.idx1 > self.start_frame - int(self.warmup * self.fps):
                        print(self.time_stamp + " - warmup")       
                    else:
                        print(self.time_stamp + " - captured")       

            else:
                self.capture.grab() 
                print(self.time_stamp)
                continue

            # =============================================================================
            # CAPTURE FRAME 
            # =============================================================================
            if capture_frame == True:              
                                
                # apply mask, if present
                if "mask_objects" in vars(self):
                
                    mask_dummy1 = np.zeros(self.frame.shape[0:2], dtype=np.int8)
                    mask_list = []
                    mask_label_names = []
                    
                    for obj in self.mask_objects:
                        mask, label, include = obj
                        if include == True:
                            mask_list.append(mask)
                            mask_label_names.append(label)
                            mask_dummy2 = np.zeros(self.frame.shape[0:2], dtype=np.int8)
                            mask_dummy2[mask] = 1
                            mask_dummy1 = np.add(mask_dummy1, mask_dummy2)
                        if include == False:
                            mask_dummy2 = np.zeros(self.frame.shape[0:2], dtype=np.int8)
                            mask_dummy2[mask] = -100
                            mask_dummy1 = np.add(mask_dummy1, mask_dummy2)
                            
                    self.frame[mask_dummy1<=0]=0
    
                # initiate tracking    
                self.fgmask = self.fgbg_subtractor.apply(self.frame)     
                self.frame_overlay = self.frame    
                                
                # apply methods
                if "methods" in vars(self):
                    for m in self.methods:
                        self.overlay, self.method_contours, self.frame_df = m._run(frame=self.frame, fgmask=self.fgmask)
                        
                        # "shadowing of methods
                        if "consecutive" in vars(self):
                            self.method_mask = np.zeros_like(self.fgmask)
                            for contour in self.method_contours:
                                if self.consecutive[0] == "contour":
                                    self.method_mask = cv2.drawContours(self.method_mask, [contour], 0, colours.white, -1) # Draw filled contour in mask   
                                elif self.consecutive[0] == "ellipse":
                                    self.method_mask = cv2.ellipse(self.method_mask, cv2.fitEllipse(contour), colours.white, -1)                                    
                                elif self.consecutive[0] == "rectangle":
                                    rx,ry,rw,rh = cv2.boundingRect(contour)
                                    cv2.rectangle(self.method_mask,(int(rx),int(ry)),(int(rx+rw),int(ry+rh)), colours.white,-1)
                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(self.consecutive[1],self.consecutive[1]))
                                self.method_mask = cv2.dilate(self.method_mask, kernel, iterations = 1)   
                            self.fgmask = cv2.subtract(self.fgmask, self.method_mask)
                        
                        # create overlay for each method
                        self.frame_overlay = cv2.addWeighted(self.frame_overlay, 1, self.overlay, weight, 0)    
                        
                        # make data.frame
                        self.frame_df.insert(0, 'frame_abs',  self.idx1)
                        self.frame_df.insert(1, 'frame',  int((self.idx1-self.start_frame)/self.skip))
                        self.frame_df.insert(2, 'mins',  mins)
                        self.frame_df.insert(3, 'secs',  secs)
                        self.df = self.df.append(self.frame_df, ignore_index=True, sort=False)                     
                    
                    # =============================================================================
                    # SAVE
                    # =============================================================================
                    
                    # show output
                    if show_selector == "overlay":
                        self.show = self.frame_overlay
                    elif show_selector == "fgmask":
                        self.show = self.fgmask    
                    else:
                        self.show = self.frame
                        
                    self.show = cv2.resize(self.show, (0,0), fx=self.resize_factor, fy=self.resize_factor)       
                    if show == True:
                        cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
                        cv2.imshow('phenopype', self.show)   
                                    
                    # save output
                    if self.save_video == True:
                        if self.save_colour and len(self.show.shape)<3:
                            self.show = cv2.cvtColor(self.show, cv2.COLOR_GRAY2BGR)
                        self.writer.write(self.show)

            # keep stream open
            if cv2.waitKey(1) & 0xff == 27:
                self.capture.release()
                self.writer.release()
                break

        # cleanup
        self.capture.release()
        self.writer.release()
        cv2.destroyAllWindows()
        
        if return_df == True:
            return self.df
        
        
class tracking_method():
    
    """Constructs a tracking method that can be supplied to the tracker. 
    
    Parameters
    ----------
    mode: str (default: "multiple")
        how many objects to track: "multiple", or "single" (biggest by diameter) objects
    operations: list (default: ["diameter", "area"])
        determines the type of operations to be performed on the detected objects:
            - "diameter" of the bounding circle of our object
            - "area" within the contour of our object
            - "grayscale" mean and standard deviation of grayscale pixel values inside the object contours
            - "grayscale_background" background within boundingbox of contour
            - "bgr" mean and standard deviation of blue, green and red pixel values inside the object contours
    blur: tuple
        blurring of fgbg-mask (kernel size, threshold [1-255])
    min_length: int (default: 1)
        minimum length in pixels for objects to be included
    remove_shadows: bool
        remove shadows if shadow-detection is actived in MOG-algorithm
    mask: list
        phenoype mask-objects (lists of boolean mask, label, and include-argument) to include or exclude an area from the procedure
    overlay_colour: phenopype colour object (default: red [red, green, blue, black, white])
        which colour should tracked objects have
    exclude: ? (forgot what this does)
    """
    def __init__(self, label="default", overlay_colour="red", min_length=1, max_length=1000,
                      mode="multiple", operations=[], mask=[], exclude=True, **kwargs):
        
        self.label = label
        self.overlay_colour = overlay_colour
        self.min_length = min_length
        self.max_length = max_length
        self.mode = mode 
        self.operations = operations
        self.mask = mask
        self.exclude = exclude

        for key, value in kwargs.items():
            if key in kwargs:
                setattr(self, key, value)


    def _print_settings(self, width=30, indent=1, compact=True):
        """Prints the settings of the tracking method. Internal reference - don't call this directly. 
        """ 

        pretty = pprint.PrettyPrinter(width=width, compact=compact, indent=indent)
        pretty.pprint(vars(self))
        
            
    def _run(self, frame, fgmask, **kwargs):
        """Run tracking method on current frame. Internal reference - don't call this directly.      
           => needs better documentation / code-referencing - I already forgot how it works.
        """
        self.frame = frame
        self.fgmask = fgmask
        self.overlay = np.zeros_like(self.frame) 
        self.overlay_bin = np.zeros(self.frame.shape[0:2], dtype=np.uint8) 
        self.frame_df = pd.DataFrame()

        if "remove_shadows" in vars(self):
            if self.remove_shadows==True:
                ret, self.fgmask = cv2.threshold(self.fgmask, 128, 255, cv2.THRESH_BINARY)
         
        if "blur" in vars(self):
            blur_kernel, thresh_val = (self.blur)
            self.fgmask = blur(self.fgmask, blur_kernel)            
            
        if "mask_objects" in vars(self):          
            mask_dummy1 = np.zeros(self.frame.shape[0:2], dtype=bool)
            mask_list = []
            mask_label_names = []
                        
            for obj in self.mask_objects:
                mask, label, include = obj
                if include == True:
                    mask_list.append(mask)
                    mask_label_names.append(label)
                    mask_dummy2 = np.zeros(self.frame.shape[0:2], dtype=np.uint8)
                    mask_dummy2[mask] = 1
                    mask_dummy1 = np.add(mask_dummy1, mask_dummy2)
                if include == False:
                    mask_dummy2 = np.zeros(self.frame.shape[0:2], dtype=np.uint8)
                    mask_dummy2[mask] = -100
                    mask_dummy1 = np.add(mask_dummy1, mask_dummy2)
                    
            mask_dummy1[mask_dummy1<0]=0
            mask_dummy1[mask_dummy1>0]=255
            
            self.mask = mask_dummy1
            
            if self.exclude==True:
                self.fgmask = np.bitwise_and(self.mask, self.fgmask)
        
        # =============================================================================
        # find objects
        # =============================================================================
        
        ret, contours, hierarchy = cv2.findContours(self.fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1) #CHAIN_APPROX_NONE
        
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

            if len(list_contours) > 0:      
                # if single biggest contour:
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
                list_mask_check = []
                
                for contour, coordinate in zip(list_contours, list_coordinates):
                    
                    # operations    
                    x=int(coordinate[0])
                    y=int(coordinate[1])
                    list_x.append(x)
                    list_y.append(y)
                    
                    if "mask_objects" in vars(self):
                        temp_list = []
                        for i in mask_list:
                            temp_list.append(i[y,x])                        
                        list_mask_check.append(temp_list)

                    rx,ry,rw,rh = cv2.boundingRect(contour)
                    frame_roi = self.frame[ry:ry+rh,rx:rx+rw]
                    frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                    mask_roi = self.fgmask[ry:ry+rh,rx:rx+rw]               
                    
                    if any("area" in o for o in self.operations):
                        list_area.append(int(cv2.contourArea(contour)))
            
                    if any("grayscale" in o for o in self.operations):
                        grayscale = ma.array(data=frame_roi_gray, mask = np.logical_not(mask_roi))
                        list_grayscale.append(int(np.mean(grayscale)))
                        
                    if any("grayscale_background" in o for o in self.operations):
                        grayscale_background = ma.array(data=frame_roi_gray, mask = mask_roi)
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
                    self.overlay = cv2.drawContours(self.overlay, [contour], 0, self.overlay_colour, -1) # Draw filled contour in mask     
                    self.overlay = cv2.putText(self.overlay, self.label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,  1,self.overlay_colour,1,cv2.LINE_AA)  
                    self.overlay = cv2.rectangle(self.overlay,(rx,ry),(int(rx+rw),int(ry+rh)),self.overlay_colour,2)
    
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
                self.frame_df = frame_df
                
                if "mask_objects" in vars(self):
                    mask_df = pd.DataFrame(list_mask_check, columns=mask_label_names)
                    self.frame_df = pd.concat([frame_df.reset_index(drop=True), mask_df], axis=1)
                    
                self.contours = list_contours
                          
                return self.overlay, self.contours, self.frame_df

            else:
                frame_df = pd.DataFrame()
                return self.overlay, [], self.frame_df
        
        else:
            frame_df = pd.DataFrame()
            return self.overlay, [], self.frame_df
        
        
        

