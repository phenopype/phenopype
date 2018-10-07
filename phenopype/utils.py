# -*- coding: utf-8 -*-
"""
Last Update: 2018/10/02
Version 0.4.0
@author: Moritz Lürig
"""

import cv2
import numpy as np
from PIL import Image
from collections import Counter

#%%
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
black = (0,0,0)
white = (255,255,255)

#%% helper functions

def exif_date(path): 
       date = Image.open(path)._getexif()[36867]
       date = date[0:4] + "-" + date[5:7] + "-" + date[8:10]
       return date

def blur(image, blur_kern):
    kern = np.ones((blur_kern,blur_kern))/(blur_kern**2)
    ddepth = -1
    return cv2.filter2D(image,ddepth,kern)

def gray_scale(image, resize=0.25):
    img = image
    if resize:
        img = cv2.resize(img, (0,0), fx=1*resize, fy=1*resize) 
    vec = np.ravel(img)
    mc = Counter(vec).most_common(9)
    g = [item[0] for item in mc]
    return int(np.median(g))

