# -*- coding: utf-8 -*-
"""
Created: 2016/03/31
Last Update: 2018/10/02
Version 0.3
@author: Moritz Lürig
"""

#%% STARTUP
import os
import sys
import cv2
sys.path.append(os.getcwd())

os.chdir("E:\\python1\\phenopype")
import phenopype as pp

from phenopype import custom

# set working, image, and output directories
in_dir = "E:\\python1\\iso_cv\\sandbox\\raw3"
out_dir = "E:\\python1\\iso_cv\\sandbox\\out_dir_kim"

#%% OUTSIDE LOOP - do only once

# =============================================================================
# SCALE chose any image to pick scale, draw it and automatically save it for 
# future reference (e.g. using scale.find, see below)
# =============================================================================
example_image = os.path.join(in_dir, os.listdir(in_dir)[0])
scale = pp.scale_maker()
scale.original, scale.mask = scale.draw(example_image, length=10, unit="mm", zoom=True)

# =============================================================================
# GRAYSCALE ADJUSTMENT - find average grayscale values in all or in a subset 
# of images, then correct to this value using gray_value_ref in object finder
# =============================================================================
# grayscale_finder(source=images_dir, mode="dir", resize=0.1)

## leave outside loop to draw arena only once
#arena = poly_drawer()
#arena.draw(image)

#%% INSIDE LOOP - do every time
index = 0

for filename in os.listdir(in_dir):
    while(1):
        # =============================================================================
        # FLOW CONTROL - continue with last index OR skip all images in out_dir
        # =============================================================================
        # if os.listdir(in_dir).index(filename) < index:
        #     continue
    #    if filename in os.listdir(out_dir):
    #        continue
        path = os.path.join(in_dir, filename)
    
        # =============================================================================
        # HELPER FUNCTIONS - for information run help(pp.scale) or help(pp.poly_drawer)
        # scale.find finds drawn scale in image, poly_drawer draws arena in every image
        # =============================================================================
        scale.current, scale.mask = scale.find(path, resize=0.25) 
        arena = pp.polygon_maker()
        arena.draw(path, show=True)
    
        # =============================================================================
        # OBJECT FINDER - for information run help(pp.object_finder) returns 
        # dataframe (of.df) containing results, and image (of.drawn) to check
        # =============================================================================
        of = pp.object_finder()
        of.run(im_path=path, thresholding="adaptive", min_size=40, min_area=4000, blur=35, corr_factor=-10, scale=scale.current, gray_value_ref=200, exclude = [arena.mask, scale.mask], show=True) 
    
        # =============================================================================
        # CUSTOM MODULES - can be used to improve results, append df and drawn image
        # 1) Kim's module, use help(kims_module) for more info
        # afterwards: save to specified directory
        # =============================================================================
        of.df, of.drawn = pp.kims_module(image=of.image, df=of.df, compare_with=of.drawn, thresholding = "otsu", blur=21, erosion=3.5, dilation=1, show=True)
        of.save(image=of.drawn, df=of.df, out_dir=out_dir, overwrite=True)
    
        # comment out of you don't want to wait after one iterations
        index += 1


#%% BREAK ALL WINDOWS
cv2.destroyAllWindows()
