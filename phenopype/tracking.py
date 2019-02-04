from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import numpy as np
import cv2
import math
import os

from phenopype.utils import (blur)


def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])
              
       
class video(object):
    def get_properties(self, video_path, **kwargs):    
        
        self.video_path = video_path       
        self.name = os.path.basename(self.video_path)


        if isinstance(self.video_path, str):
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
        else:
            print("Error - str path to video needed")
            
        self.nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.fourcc_str = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        self.length = str(int(( self.nframes / self.fps)/60)).zfill(2) + ":" +str(int((((self.nframes / self.fps)/60)-int((self.nframes / self.fps)/60))*60)).zfill(2)
        self.dimensions = frame.shape[0:2]
        if frame.shape[2] == 3:
            self.is_colour_video = True
        
        print("\n")
        print("----------------------------------------------------------------")
        print("Video properties of \"" + self.name + "\":\n")
        print("Frames per second: " + str(self.fps) + "\nN frames: " + str(self.nframes) + "\nLength: " + str(self.length) + " (mm:ss)" + "\nDimensions: " + str(self.dimensions) + "\nColour video: " + str(self.is_colour_video) + "\nFourCC code: " + str(self.fourcc_str)) 
        print("----------------------------------------------------------------")


    def tracking_setup(self, **kwargs):
        
        name_nosuf = os.path.splitext(os.path.basename(self.video_path))[0]
        suffix = kwargs.get("suffix", "out")
        name_out = kwargs.get("name", name_nosuf + "_" + suffix + ".avi")  
        dir_out = kwargs.get("save_dir", os.path.dirname(self.video_path))
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        self.path_out = os.path.join(dir_out, name_out)
            
        fps_out = kwargs.get("fps", self.fps)
        dim_out = kwargs.get("dimensions", tuple(reversed(self.dimensions)))
        self.fourcc_str_out = kwargs.get("fourcc_string", "DIVX")
        self.fourcc_out = cv2.VideoWriter_fourcc(*self.fourcc_str_out)
        self.save_as_colour = kwargs.get("save_as_colour", self.is_colour_video)

        history = kwargs.get("history", 60)
        backgr_thresh = kwargs.get("backgr_thresh", 10)        
        detect_shadows = kwargs.get("detect_shadows", True)
        
        self.skip = kwargs.get("skip", 5)
        self.wait = kwargs.get("start_after", 15)
        if "finish_after" in kwargs:
            self.finish = kwargs.get("finish_after")

        self.writer = cv2.VideoWriter(self.path_out, self.fourcc_out, fps_out, dim_out, self.save_as_colour) 
        self.fgbg_subtractor = cv2.createBackgroundSubtractorMOG2(history = int(history * (self.fps / self.skip)), varThreshold = int(backgr_thresh), detectShadows = detect_shadows)

        print("\n")
        print("----------------------------------------------------------------")
        print("Settings for motion tracking in \"" + self.name + "\":\n")
        print("Save name: " + name_out + "\nSave path: " + os.path.dirname(self.video_path) + "\nFrames per second: " + str(fps_out) + "\nDimensions: " + str(dim_out) + "\nFormat (FourCC code) : " + self.fourcc_str_out + "\n\"History\"-parameter: " + str(history) + " seconds" + "\nSensitivity: " + str(backgr_thresh) + "\nRead every nth frame: " + str(self.skip) + "\nDetect shadows: " + str(detect_shadows) + "\nStart after n seconds: " + str(self.wait) + "\nFinish after n seconds: " + str(self.finish if self.finish else " - ")) 
        print("----------------------------------------------------------------")
        
    def tracking(self, **kwargs):
        
        self.idx1, self.idx2 = (0,0)
        self.capture = cv2.VideoCapture(self.video_path)        
        self.wait_frame = self.wait * self.fps
        if self.finish:
            self.finish_frame = self.finish * self.fps

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
                print(self.time_stamp + " - captured")
            else:
                print(self.time_stamp)
                
                
            # video writer  
            if self.capture_frame == True:    

                cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
                cv2.imshow('phenopype', self.fgmask)    
                self.writer.write(self.fgmask)
            
            # keep stream open
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # cleanup
        self.capture.release()
        self.writer.release()
        cv2.destroyAllWindows()
        
        
#class test_module(object):
#    def track(**kwargs):
#        fgmask = video.fgmask
#        if "blur" in kwargs:
#            blur_kernel, thresh_val = kwargs.get("blur",[9, 127])
#            blurred = blur(fgmask, blur_kernel)

 

class motion_tracker(object):
    def __init__(self, frame, fg_mask, min_length, **kwargs):
        
        shadows = kwargs.get("detect_shadows", True)

        if shadows == True:
            ret, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        
        if "blur" in kwargs:
            blur_kernel, thresh_val = kwargs.get("blur",[9, 127])
            self.blurred = blur(fg_mask, blur_kernel)
            ret, self.thresh = cv2.threshold(self.blurred, thresh_val, 255, cv2.THRESH_BINARY)
            
        
        ret, contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1) #CHAIN_APPROX_NONE

        # fish ellipse, length, and center
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
            
            # check against min length
            if ellipses_l[np.argmax(ellipses_l)] > min_length:
                
                self.contour = contours[np.argmax(ellipses_l)]
                self.ellipse = ellipses[np.argmax(ellipses_l)]
                self.center = np.array([self.ellipse[0]])
    
                # draw fish mask
                self.mask = np.zeros_like(frame) # Create mask where white is what we want, black otherwise
                self.contour_drawn = cv2.drawContours(self.mask, [self.contour], 0,(0,0, 255), -1) # Draw filled contour in mask     
                
                # make extended box  to mask for isopod detection
                box = cv2.minAreaRect(self.contour)
                box = tuple([box[0], (box[1][0] + 200, box[1][1] + 150), box[2]])
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.box = np.full_like(gray, 255)
                self.box = cv2.drawContours(self.box,[np.int0(cv2.boxPoints(box))],0,0,-1)
                               
                self.empty = False
                
            else:
                self.empty = True
        else: 
            self.empty = True
            

        frame_out = frame
        frame_out = cv2.addWeighted(frame_out, 1, self.contour_drawn, 0.5, 0)
        frame_out = cv2.ellipse(frame_out, tuple(self.ellipse[0:3]), (0,0,255), 3)
        frame_out = cv2.putText(frame_out, "Fish", (int(self.center[0,0]), int(self.center[0,1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)  


        

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