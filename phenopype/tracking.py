from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import numpy as np
import cv2
import math
import copy
import os

from phenopype.utils import (blur)


def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]


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
            
    def draw(self, frame, arguments):
        frame_out = frame
        
        if not self.empty:
            if "contour" in arguments:
                frame_out = cv2.addWeighted(frame_out, 1, self.contour_drawn, 0.5, 0)
            if "ellipse" in arguments:
                frame_out = cv2.ellipse(frame_out, tuple(self.ellipse[0:3]), (0,0,255), 3)
            if "text" in arguments:
                frame_out = cv2.putText(frame_out, "Fish", (int(self.center[0,0]), int(self.center[0,1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)  
            return frame_out
        
        else:
            return frame_out

                 
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
       
class video(object):
    def get_info(self, video_path):    
        
        self.video_path = video_path
        if isinstance(self.video_path, str):
            cap = cv2.VideoCapture(self.video_path)
        else:
            print("Error - str path to video needed")
            
        self.nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.length = str(int(( self.nframes / self.fps)/60)).zfill(2) + ":" +str(int((((self.nframes / self.fps)/60)-int((self.nframes / self.fps)/60))*60)).zfill(2)

        idx = 0
        while(cap.isOpened()):
            idx = idx + 1
            # extract frame from video stream
            ret, frame = cap.read()
            if idx == 5:
                break
            
        self.name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.frame = frame
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.layers = frame.shape[2]

    def io(self, video_format="DIVX", backgr_thresh=10, **kwargs):  
        
        shadows = kwargs.get("detect_shadows", True)
        our_dir = kwargs.get("save_dir", os.path.dirname(self.video_path))
        suffix = kwargs.get("suffix", "out")
        
        history = kwargs.get("history", 1)
        backgr_thresh = kwargs.get("backgr_thresh", 10)        
        skip = kwargs.get("skip", 4)

        fourcc = cv2.VideoWriter_fourcc(*video_format)
        self.writer = cv2.VideoWriter(os.path.join(our_dir,  self.name + "_" + suffix + ".avi"), fourcc, self.fps, (self.width, self.height), True) 
        self.fgbg_subtractor = cv2.createBackgroundSubtractorMOG2(history = int((history*60) * (self.fps / skip)), varThreshold = int(backgr_thresh), detectShadows = shadows)

    def tracking(self):
        idx1, idx2 = (0,0)
        df_fish, df_isopod  = (pd.DataFrame(),pd.DataFrame())
        cap = cv2.VideoCapture(video_path)

            
    
        
class video_time(object):
    def __init__(self, idx, fps):
        self.mins = str(int((idx / fps)/60)).zfill(2)
        self.secs = str(int((((idx / fps)/60)-int(self.mins))*60)).zfill(2)
        
def capture_feedback(mins, secs, index, length, nframes, capture):       
    if capture == True:
        print(mins + ":" + secs + " / " + length + " - " + str(index) + " / " + str(int(nframes)) + " - captured")
    else:
        print(mins + ":" + secs + " / " + length + " - " + str(index) + " / " + str(int(nframes)))

class polygon_drawer(object):
    
    # initiate
    def __init__(self, video_name, save_path):
        self.window_name = video_name # Name for our window
        self.video_name = video_name
        self.save_path = save_path # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        
        self.FINAL_LINE_COLOR = (0, 255, 0)
        self.FILL_COLOR = (255,255,255)
        self.WORKING_LINE_COLOR = (255, 0, 0)

    # mouse action
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    # draw lines
    def run(self, image):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        temp_canvas = copy.deepcopy(image)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(temp_canvas, np.array([self.points]), False, self.FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(temp_canvas, self.points[-1], self.current, self.WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, temp_canvas)
            temp_canvas = copy.deepcopy(image)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # create final arena
        canvas = image
        zeros = np.zeros(canvas.shape, np.uint8)
        red = np.zeros(canvas.shape, np.uint8)
        red[:,:,2] = 255
        
        if (len(self.points) > 0):
            cv2.fillPoly(red, np.array([self.points]), self.FINAL_LINE_COLOR)
               
        if (len(self.points) > 0):
            cv2.fillPoly(zeros, np.array([self.points]), self.FILL_COLOR)
            
        # return for further use
        self.mask_colour = zeros
        self.mask_gray = cv2.cvtColor(zeros, cv2.COLOR_BGR2GRAY)
        self.image = cv2.addWeighted(copy.deepcopy(canvas), .8, red, 0.2, 0)
        self.points = np.array([self.points])
        self.rect = cv2.boundingRect(self.points)

        # Waiting for the user to press any key and return points
        cv2.imshow(self.window_name, self.image)
        save_image_path = os.path.join(self.save_path, self.video_name + "_arena.png")
        cv2.imwrite(save_image_path, self.image)    
        
        # show image of polygon
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        print("Arena image saved: " + save_image_path)

  