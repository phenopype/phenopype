import os
import cv2
import numpy as np
import exifread
from collections import Counter

#%% colours

green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
black = (0,0,0)
white = (255,255,255)


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
    """
    out_dir = save_dir     
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if "append" in kwargs:
        app = '_' + kwargs.get('append',"objects")
    else:
        app = ""
        
    df_path=os.path.join(out_dir , name +  app + ".txt")
    df = df.fillna(-9999)
    df = df.astype(str)
    if kwargs.get('overwrite',True) == False:
        if not os.path.exists(df_path):
            df.to_csv(path_or_buf=df_path, sep=",")

    else:
            df.to_csv(path_or_buf=df_path, sep=",")


def save_img(image, name, save_dir, **kwargs):
    """Save an image (array) to jpg.
    """
    # set dir and names
    out_dir = save_dir     
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if "append" in kwargs:
        app = '_' + kwargs.get('append',"processed")
    else:
        app = ""
        
    im_path=os.path.join(out_dir , name +  app + ".jpg")
    
    if "resize" in kwargs:
        factor = kwargs.get('resize')
        image = cv2.resize(image, (0,0), fx=1*factor, fy=1*factor) 
    
    if kwargs.get('overwrite',True) == False:
        if not os.path.exists(im_path):
            cv2.imwrite(im_path, image)
    else:
        cv2.imwrite(im_path, image)


