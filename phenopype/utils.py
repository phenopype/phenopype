import cv2
import numpy as np
import exifread
import collections as col

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

def gray_scale(source, **kwargs):
    img = source
    if "resize" in kwargs:
        img = cv2.resize(img, (0,0), fx=1*kwargs.get("resize"), fy=1*kwargs.get("resize")) 
    vec = np.ravel(img)
    mc = col.Counter(vec).most_common(9)
    g = [item[0] for item in mc]
    return int(np.median(g))

def find_skeleton3(img):
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

        
        
        