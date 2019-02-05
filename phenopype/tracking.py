from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import numpy as np
import cv2
import math
import os
import pprint

from phenopype.utils import (blur)



def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])
              
       
class motion_tracker(object):
    def __init__(self, video_path, **kwargs): 
        """Read properties of input video file.
        """
        if isinstance(video_path, str):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            

        else:
            print("Error - full or relative path of video-file needed")
        
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
        
        self.writer = cv2.VideoWriter(path_out, fourcc_out, fps_out, dimensions_out, save_colour) 


        print("\n")
        print("----------------------------------------------------------------")
        print("Output video settings - \"" + self.name + "\":\n")
        print("Save name: " + name_out + "\nSave dir: " + dir_out + "\nFrames per second: " + str(fps_out) + "\nDimensions: " + str(dimensions_out) + res_msg +  "\nColour video: " + str(save_colour) + "\nFormat (FourCC code): " + fourcc_str_out) 
        print("----------------------------------------------------------------")
        
    def motion_detection(self, **kwargs):

        self.skip = kwargs.get("skip", 5)
        self.wait = kwargs.get("start_after", 15)
        if "finish_after" in kwargs:
            self.finish = kwargs.get("finish_after")
        history = kwargs.get("history", 60)
        backgr_thresh = kwargs.get("backgr_thresh", 10)        
        self.detect_shadows = kwargs.get("detect_shadows", True)

        self.fgbg_subtractor = cv2.createBackgroundSubtractorMOG2(history = int(history * (self.fps / self.skip)), varThreshold = int(backgr_thresh), detectShadows = self.detect_shadows)
        
        if "methods" in kwargs:
            self.methods = kwargs.get("methods")
            for m in self.methods:
                m.print_settings()
        
        print("\n")
        print("----------------------------------------------------------------")
        print("Motion detection settings - \"" + self.name + "\":\n")
        print("\n\"History\"-parameter: " + str(history) + " seconds" + "\nSensitivity: " + str(backgr_thresh) + "\nRead every nth frame: " + str(self.skip) + "\nDetect shadows: " + str(self.detect_shadows) + "\nStart after n seconds: " + str(self.wait) + "\nFinish after n seconds: " + str(self.finish if self.finish else " - ")) 
        print("----------------------------------------------------------------")
              
         
        
        
    def run_tracking(self, **kwargs):
                
        self.idx1, self.idx2 = (0,0)
        self.capture = cv2.VideoCapture(self.path)        
        self.wait_frame = self.wait * self.fps
        if self.finish:
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
            if self.finish_frame:
                if self.idx1 >= self.finish_frame:
                    break
                             
                
            # warmup fgbg-algorithm shortly before capturing 
            if self.idx1 > self.wait_frame - (3*self.fps) and self.idx2 == 0:
                self.fgmask = self.fgbg_subtractor.apply(self.frame)               
                
            # start modules after x seconds
            if self.idx1 > self.wait_frame and self.idx2 == 0:
                self.capture_frame = True
                           
            # feedback
            self.mins = str(int((self.idx1 / self.fps)/60)).zfill(2)
            self.secs = str(int((((self.idx1 / self.fps)/60)-int(self.mins))*60)).zfill(2)    
            self.time_stamp = "Time: " + self.mins + ":" + self.secs + "/" + self.length + " - Frames: " + str(self.idx1) + "/" + str(int(self.nframes))


                
            if self.capture_frame == True:    
                for m in self.methods:
                    self.show = m._run(self.frame, self.fgmask, self.resize_factor)
                print(self.time_stamp + " - captured")
                
                self.writer.write(self.show)
                cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
                cv2.imshow('phenopype', self.show)    
                
            else:
                print(self.time_stamp)

            # keep stream open
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # cleanup
        self.capture.release()
        self.writer.release()
        cv2.destroyAllWindows()
        
        
class tracking_method(object):
    def __init__(self, **kwargs):
        """Constructs a tracking method that can be supplied to the tracker. 
        """                   
        for key, value in kwargs.items():
            if key in kwargs:
                setattr(self, key, value)
            
    def print_settings(self, **kwargs):
        """Prints the settings of the tracking method. 
        """ 
        width = kwargs.get("width",30)
        indent = kwargs.get("indent",1)
        compact = kwargs.get("compact",True)
        pretty = pprint.PrettyPrinter(width=width, compact=compact, indent=indent)
        
        pretty.pprint(vars(self))
        
            
    def _run(self, frame_in, fgmask, resize_factor, **kwargs):
        """Run tracking method on current frame. Internal reference - don't call this directly.
        """
        
        self.frame_out = frame_in
        self.fgmask = fgmask
        
        if "remove_shadows" in vars(self):
            if self.remove_shadows==True:
                ret, self.mask = cv2.threshold(self.fgmask, 128, 255, cv2.THRESH_BINARY)
         
        if "blur" in vars(self):
            blur_kernel, thresh_val = (self.blur)
            self.fgmask = blur(self.fgmask, blur_kernel)            
            ret, self.fgmask = cv2.threshold(self.fgmask, thresh_val, 255, cv2.THRESH_BINARY)
                    


        # =============================================================================
        # find objects
        # =============================================================================
        
        ret, contours, hierarchy = cv2.findContours(self.fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1) #CHAIN_APPROX_NONE

        # turn blobs to ellipses
        if contours:
            ellipses = []
            for cnt in contours:
                if cnt.shape[0] > 4:
                    center,axes,orientation = cv2.fitEllipse(cnt)
                    length = np.mean([math.sqrt(axes[1]*axes[0]*math.pi),max(axes)])
                    ellipses.append([center,axes,orientation,length])   
                else:
                    ellipses.append([0,0,0,0])
            ellipses_l = [l[3] for l in ellipses] 
            
        # check ellipses against min length
        if "min_length" in vars(self):
            if ellipses_l[np.argmax(ellipses_l)] < self.min_length:
                self.frame_out = cv2.putText(self.frame_out, "too small", (int(self.frame_out.shape[0]/20),int(self.frame_out.shape[1]/20)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)                
                return self.frame_out
        else:
            if len(ellipses_l) > 0:
            
                self.contour = contours[np.argmax(ellipses_l)]
                self.ellipse = ellipses[np.argmax(ellipses_l)]
                self.center = np.array([self.ellipse[0]])
    
                # draw fish mask
                self.mask = np.zeros_like(self.frame_out) # Create mask where white is what we want, black otherwise
                self.contour_drawn = cv2.drawContours(self.mask, [self.contour], 0,(0,0, 255), -1) # Draw filled contour in mask     
                
                # make extended box to mask for isopod detection
                box = cv2.minAreaRect(self.contour)
                box = tuple([box[0], (box[1][0] + 200, box[1][1] + 150), box[2]])
                
                gray = cv2.cvtColor(self.frame_out, cv2.COLOR_BGR2GRAY)
                self.box = np.full_like(gray, 255)
                self.box = cv2.drawContours(self.box,[np.int0(cv2.boxPoints(box))],0,0,-1)
    
                self.frame_out = cv2.addWeighted(self.frame_out, 1, self.contour_drawn, 0.5, 0)
                self.frame_out = cv2.ellipse(self.frame_out, tuple(self.ellipse[0:3]), (0,0,255), 3)
                self.frame_out = cv2.putText(self.frame_out, self.label, (int(self.center[0,0]), int(self.center[0,1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)  
        
                self.frame_out = cv2.resize(self.frame_out, (0,0), fx=1*resize_factor, fy=1*resize_factor) 

                return self.frame_out
            
            else:
                
                self.frame_out = cv2.putText(self.frame_out, "no " + self.label + " found", (int(self.frame_out.shape[0]/10),int(self.frame_out.shape[1]/10)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)   
                return self.frame_out
        
        

        

class isopod_module(object):
    def __init__(self, frame, fgmask, shadows, blur_kern, blur_thresh, min_length, max_length, arena_mask):
        
        # crop non-arena-area
        fgmask = np.bitwise_and(fgmask, arena_mask)
            
        if shadows == True:
            ret, fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        
        # fish contours and centerpoint
        self.blurred = blur(fgmask, blur_kern)
        ret, self.thresh = cv2.threshold( self.blurred, blur_thresh, 255, cv2.THRESH_BINARY)
        ret, contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #CHAIN_APPROX_TC89_L1
    
        # isopod ellipse, length, and center
        if contours:
            self.ellipses = [] 
            self.contours_good = [] 
            self.contours_noise = []
            for cnt in contours:
                if cnt.shape[0] > 4:
                    center,axes,orientation = cv2.fitEllipse(cnt)
                    length = np.mean([math.sqrt(axes[1]*axes[0]*math.pi),max(axes)])
                    if length < max_length and length > min_length:
                        self.ellipses.append([center,axes,orientation,length]) #self.
                        self.contours_good.append(cnt) #self.
                    else:
                        self.contours_noise.append(cnt)
             
                    self.center = [ellipse[0] for ellipse in self.ellipses]
                    
                    self.contour_drawn = np.zeros_like(frame) # Create mask where white is what we want, black otherwise
                    for contour in self.contours_good:
                        self.contour_drawn = cv2.drawContours(self.contour_drawn, [contour], 0,(0,255, 0), -1) # Draw filled contour in mask                
            if len(self.contours_good) > 0:
                self.empty = False   
            else:
                self.empty = True
        else:
            self.empty = True
            
            
    def draw(self, frame, arguments):
        frame_out = frame
        
        if not self.empty:
            if "contour" in arguments:
                frame_out = cv2.addWeighted(frame_out, 1, self.contour_drawn, 0.5, 0)
            if "ellipse" in arguments:
                for ellipse in self.ellipses:   
                    frame_out = cv2.ellipse(frame_out, tuple(ellipse[0:3]), (0,255,0), 1)
            if "text" in arguments:
                for center in self.center:    
                    frame_out = cv2.putText(frame_out, "Isopod", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX,  1,(0,255,0),1,cv2.LINE_AA)  
            return frame_out
        
        else:
            return frame_out          