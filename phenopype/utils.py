import os
import cv2
import copy
import numpy as np
import exifread
from collections import Counter

#%% colours

green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
black = (0,0,0)
white = (255,255,255)

colours = {"red": (0, 0, 255),
 "green": (0, 255, 0), 
 "blue": (255, 0, 0),
 "black":(0,0,0),
 "white":(255,255,255)}


#%% modules

def exif_date(path): 
    f = open(path, 'rb')
    tags = exifread.process_file(f)
    t = str(tags["EXIF DateTimeOriginal"])
    return t[0:4] + "-" + t[5:7] + "-" + t[8:10] + " " + t[11:20]

def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]

def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

def blur(image, blur_kern):
    kern = np.ones((blur_kern,blur_kern))/(blur_kern**2)
    ddepth = -1
    return cv2.filter2D(image,ddepth,kern)

def find_skeleton(img):
    skeleton = np.zeros(img.shape,np.uint8)
    eroded = np.zeros(img.shape,np.uint8)
    temp = np.zeros(img.shape,np.uint8)

    _,thresh = cv2.threshold(img,127,255,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton,iters)
        
        
def find_centroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return int(sum_y/length), int(sum_x/length)

def show_img(img):
    if isinstance(img, str):
        image = cv2.imread(img)  
        cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
        cv2.imshow('phenopype', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif isinstance(img, list):
        idx=0
        for i in img:
            idx+=1
            cv2.namedWindow('phenopype' + " - " + str(idx) ,cv2.WINDOW_NORMAL)
            cv2.imshow('phenopype' + " - " + str(idx), i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        image = img
        cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
        cv2.imshow('phenopype', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
#%% grayscale
    
    
def get_median_grayscale(image, **kwargs):
    if (image.shape[0] + image.shape[1])/2 > 2000:
        factor = kwargs.get('resize', 0.5)
        image = cv2.resize(image, (0,0), fx=1*factor, fy=1*factor) 
        
    vector = np.ravel(image)
    vector_mc = Counter(vector).most_common(9)
    g = [item[0] for item in vector_mc]
    return int(np.median(g))
    
#        def project_grayscale_finder(self, **kwargs):
#        """Returns median grayscale value from all images inside the project image directory.
#        
#        Parameters
#        -----------
#        
#        resize: in (0.1-1)
#            resize image to increase speed 
#        write: bool, default False
#            write median grayscale to project dataframe
#            
#        """
#        
#        write = kwargs.get('write', False)
#
#        
#        self.gray_scale_list = []
#        for filepath, filename in zip(self.filepaths, self.filenames):
#            image = cv2.imread(filepath,0)
#            med = get_median_grayscale(image)
#            self.gray_scale_list.append(med)
#            print(filename + ": " + str(med))
#            
#        print("\nMean grayscale in directory: " + str(int(np.mean(self.gray_scale_list))))
#        
#        if write == True:
#            self.df["gray_scale"] = self.gray_scale_list
#%% save functions
    
def save_csv(df, name, save_dir, **kwargs):
    """Save a pandas dataframe to csv. 
    
    Parameters
    ----------
    df: df
        object_finder outpur (pandas data frame) to save
    name: str
        name for saved df
    save_dir: str
        location to save df
    append: str (optional)
        append df name with string to prevent overwriting
    overwrite: bool (optional, default: False)
        overwrite df if name exists
    """
    out_dir = save_dir     
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    app = kwargs.get('append',"_results")
    new_name = os.path.splitext(name)[0] + app
        
    df_path=os.path.join(out_dir , new_name  + ".txt")
    
    df = df.fillna(-9999)
    df = df.astype(str)
    
    if kwargs.get('overwrite',True) == False:
        if not os.path.exists(df_path):
            df.to_csv(path_or_buf=df_path, sep=",")

    else:
            df.to_csv(path_or_buf=df_path, sep=",")


def save_img(image, name, save_dir, **kwargs):
    """Save an image (array) to jpg.
    
    Parameters
    ----------
    image: array
        image to save
    name: str
        name for saved image
    save_dir: str
        location to save image
    append: str ("")
        append image name with string to prevent overwriting
    extension: str ("")
        file extension to save image with
    overwrite: bool (optional, default: False)
        overwrite images if name exists
    """
    # set dir and names
    out_dir = save_dir     
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        

    app = kwargs.get('append',"")
    new_name = os.path.splitext(name)[0] + app
        
    ext = kwargs.get('extension',os.path.splitext(name)[1])
    new_name = new_name + ext
    
    im_path=os.path.join(out_dir , new_name)
    
    if "resize" in kwargs:
        factor = kwargs.get('resize')
        image = cv2.resize(image, (0,0), fx=1*factor, fy=1*factor) 
    
    if kwargs.get('overwrite',False) == False:
        if not os.path.exists(im_path):
            cv2.imwrite(im_path, image)
    else:
        cv2.imwrite(im_path, image)
       


#%% detectors
        
class image_registration:
    """Generic image registration method."
        
        Parameters
        ----------
        image: str or array
            absolute or relative path to OR numpy array of image containing the template 
        mode: str (default: "rectangle")
            mark the object with a polygon or a rectangle

        """
        
    def __init__(self, image, **kwargs):
        # initialize # ----------------
        
        if isinstance(image, str):
            image = cv2.imread(image)
        if not len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        self.image = image
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.idx = 0
        
        mode = kwargs.get("mode","rectangle")
        
        cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("phenopype", self._on_mouse)
        
        temp_canvas1 = copy.deepcopy(self.image)
        temp_canvas2 = copy.deepcopy(self.image)

        print("\nMark the outline of your mask")
  
                    
        # =============================================================================
        # draw rectangle 
        # =============================================================================             
                
        if mode == "rectangle":
            (x,y,w,h) = cv2.selectROI("phenopype", temp_canvas1, fromCenter=False)
            if cv2.waitKey(50) & 0xff == 13:
                cv2.destroyWindow("phenopype")
                self.done = True
            elif cv2.waitKey(50) & 0xff == 27:
                cv2.destroyWindow("phenopype")  
                self.done = True
            self.points = [(x, y), (x, y+h), (x+w, y+h), (x+w, y)]
            self.done = True
            
        # =============================================================================
        # draw polygon 
        # =============================================================================
        
        elif mode == "polygon":
            while(not self.done):
                if (len(self.points) > 0):
                    cv2.polylines(temp_canvas1, np.array([self.points]), False, green, 3)
                    cv2.line(temp_canvas1, self.points[-1], self.current, blue, 3)
                cv2.imshow("phenopype", temp_canvas1)
                temp_canvas1 = copy.deepcopy(temp_canvas2)
                if cv2.waitKey(50) & 0xff == 13:
                    self.done = True
                    cv2.destroyWindow("phenopype")
                elif cv2.waitKey(50) & 0xff == 27:
                    self.done = True
                    cv2.destroyWindow("phenopype")
                    
                    
        # create template image for registration
        rx,ry,w,h = cv2.boundingRect(np.array(self.points, dtype=np.int32))
        self.image_original_template = self.image[ry:ry+h,rx:rx+w]
        
        # create mask for registration
        self.mask_original_template = np.zeros(self.image.shape[0:2], np.uint8)
        cv2.fillPoly(self.mask_original_template, np.array([self.points]), white) 
        self.mask_original_template = self.mask_original_template[ry:ry+h,rx:rx+w]
                
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
    def _on_mouse(self, event, x, y, buttons, user_param):
        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.idx += 1
            print("Adding point #%d with position(%d,%d) to overlay" % (self.idx, x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points = self.points[:-1]
                self.idx -= 1
                print("Removing point #%d with position(%d,%d) from overlay" % (self.idx, x, y))
            else:
                print("No points to delete")
            
    def detect(self, image, **kwargs):
        """Find object from a defined template inside an image and update pixel ratio. Feature detection is run by the AKAZE algorithm (http://www.bmva.org/bmvc/2013/Papers/paper0013/abstract0013.pdf).  
        
        Parameters
        -----------
        image: str or array
            absolute or relative path to OR numpy array of image containing the scale 
        show: bool (optional, default: False)
            show result of scale detection procedure on current image   
        resize: num (optional, default: 1)
            resize image to speed up detection process (WARNING: too low values may result in poor detection results or even crashes)
        """
        
        # =============================================================================
        # INITIALIZE
        # =============================================================================
        
        if isinstance(image, str):
            self.image_target = cv2.imread(image)
        else:
            self.image_target = image

        image_target = self.image_target 
        image_original = self.image_original_template
        
        show = kwargs.get('show', False)
        min_matches = kwargs.get('min_matches', 10)
        
        # image diameter bigger than 2000 px
        if (image_target.shape[0] + image_target.shape[1])/2 > 2000:
            factor = kwargs.get('resize', 0.5)
        else:
            factor = kwargs.get('resize', 1)
        image_target = cv2.resize(image_target, (0,0), fx=1*factor, fy=1*factor) 
        
        if not len(image_target.shape)==3:
            image_target = cv2.cvtColor(image_target, cv2.COLOR_GRAY2BGR)
            
    
        # =============================================================================
        # SIFT detector
        # =============================================================================
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp1, des1 = sift.detectAndCompute(img1,self.mask_original_template)
        # kp2, des2 = sift.detectAndCompute(img2,None)
         
        # =============================================================================
        # ORB detector
        # =============================================================================
#        orb = cv2.ORB_create()
#        kp1, des1 = orb.detectAndCompute(img1,self.mask_original_template)
#        kp2, des2 = orb.detectAndCompute(img2,None)
#        des1 = np.asarray(des1, np.float32)
#       des2 = np.asarray(des2, np.float32)
        
#        FLANN_INDEX_KDTREE = 0
#        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#        search_params = dict(checks = 50)
#        flann = cv2.FlannBasedMatcher(index_params, search_params)
#        matches = flann.knnMatch(des1,des2,k=2)
        
        # =============================================================================
        # AKAZE detector
        # =============================================================================     
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(image_original,self.mask_original_template)
        kp2, des2 = akaze.detectAndCompute(image_target,None)       
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
            ret, contours, hierarchy = cv2.findContours(self.mask_original_template,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_TC89_L1)
            box = contours[0].astype(np.float32)

            self.rect  = cv2.perspectiveTransform(box,M).astype(np.int32)
            image_target = cv2.polylines(image_target,[self.rect],True,red,5, cv2.LINE_AA)
            
            # =============================================================================
            # compare scale to original, and return adjusted ratios
            # =============================================================================
            
            if show == True:
                cv2.namedWindow("phenopype", flags=cv2.WINDOW_NORMAL)
                cv2.imshow("phenopype", image_target)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if kwargs.get("convert",True) == True:
                self.rect = self.rect/factor
                           
            self.rect = self.rect.astype(int)
            # MASK
            zeros = np.zeros(self.image_target.shape[0:2], np.uint8)
            mask_bin = cv2.fillPoly(zeros, [np.array(self.rect)], white)       
            self.mask = np.array(mask_bin, dtype=bool)

            # TARGET SNIPPET
            (rx,ry,w,h) = cv2.boundingRect(self.rect)
            self.image_found = self.image_target[ry:ry+h,rx:rx+w]

            
            print("\n")
            print("--------------------------------------")
            print("Scale found with %d keypoint matches" % self.nkp)
            print("--------------------------------------")
            print("\n")
        
        else:
            print("\n")
            print("----------------------------------------------")
            print("Scale not found - only %d/%d keypoint matches" % (self.nkp, min_matches))
            print("----------------------------------------------")
            print("\n")
            
            return "no current scale", "no scale mask"
