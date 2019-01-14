# -*- coding: utf-8 -*-
"""
Created: 2018/10/02
Last Update: 2018/10/28
@author: Moritz Lürig
"""

#%% STARTUP

import os
import cv2
import pandas as pd
import phenopype as pp


#%% OUTSIDE LOOP - do only once

os.chdir("E:\\Python1\\phenopype\\local")
in_dir = "E:\\Python1\\phenopype\\local\\gifs\\stickles\\"
out_dir = "E:\\Python1\\phenopype\\local\\gifs\\stickles_processed\\"

proj = pp.project()    
proj.project_maker("stickle_gif", image_dir=in_dir, save_dir=out_dir, mode="dir")

scale = pp.scale_maker()
scale.grab(proj.filepaths[0], length=10, unit="mm", mode="box",resize=0.5, zoom=True)


#%% INSIDE LOOP - do for every picture

# =============================================================================
# BREAK LOOP WITH CONSOLE-STOP + "ESC" WHEN IMAGE IS SHOWING
# =============================================================================

#proj.update_list()
scale.current = scale.measured

for filepath, filename in zip(proj.filepaths, proj.filenames):
    
    
    area = pp.polygon_maker()
    area.draw(image_path=proj.filepaths[0], mode="polygon", show=True)
    
    
    of = pp.object_finder()
    of.run(im_path=proj.filepaths[0], method=("adaptive", 49,1), resize=0.5, exclude = [area.mask], scale=1, show=["image", "df"], blur1=15, label=False, min_area=200, blur2=(11,140)) 
        
    # SKIP ALREADY PROCESSED IMAGES
#    if filename in os.listdir(out_dir):
#        continue

    
    # DRAW ARENA
    arena = pp.polygon_maker()
    arena.draw(filepath, show=False, mode="box")

    # =============================================================================
    # OBJECT FINDER - for information run help(pp.object_finder) returns 
    # dataframe (of.df) containing results, and image (of.drawn) to check
    # =============================================================================
    of = pp.object_finder()
    of.run(im_path=filepath, exclude = [scale.mask, arena.mask], scale=scale.measured, show=["image", "df"], resize=0.5, blur1=9, min_diam=50, min_area=800, blur2=(19,35), corr_factor=(-5,1,"cross")) 

    # =============================================================================
    # CUSTOM MODULES - can be used to improve results, append df and drawn image
    # Kim's module, use help(pp.kims_module) for more info
    # =============================================================================
#    kim = pp.kims_module()
#    kim.run(image=of.image, df=of.df, compare_with=of.drawn, thresholding = "adaptive", sensitivity=179, roi_size=300, blur1=39, blur2=(25,180), show=True, label=True)
#    
#    # UPDATE AND SAVE IMG AND DF
#    of.df = pd.concat([of.df.reset_index(drop=False), kim.df], axis=1)
    of.save(image=of.drawn, df=of.df, save_to=out_dir, overwrite=True)


#%% destroy windows            

cv2.destroyAllWindows()
