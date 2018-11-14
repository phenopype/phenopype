# -*- coding: utf-8 -*-
"""
Last Update: 2018/10/07
Version 0.4.5
@author: Moritz Lürig
"""

#%% load modules

import os
import cv2
import phenopype as pp

#%% DEV_startup

import os
import cv2
os.chdir("E:\\python1\\phenopype")

import phenopype as pp
import importlib 
importlib.reload(pp)
pp.__file__

#%% example 1 - multiple object in one image

in_dir = os.getcwd() + "\\example\\example1\\ex1_images"

proj = pp.project()    
proj.project_maker("example1", in_dir=in_dir)

scale = pp.scale_maker()
scale.measure(proj.filepaths[0], length=10, unit="mm", crop=True, zoom=True, save=True)

# initialize object finder
of = pp.standard_object_finder()

# run with no finetuning
of.run(im_path=proj.filepaths[0], exclude = [scale.mask], scale=scale.measured, show_img=True, show_df=True) 

# with some finetuning
of.run(im_path=proj.filepaths[0], exclude = [scale.mask], scale=scale.measured, show_img=True, show_df=True, blur1=25, min_size=50, min_area=5000) 

# for comparison: with different method
of.run(im_path=proj.filepaths[0], exclude = [scale.mask], scale=scale.measured, show_img=True, show_df=True, blur1=39, min_size=50, min_area=5000, method = "adaptive", sensitivity=149) 

# default method ("otsu"), with some more finetuning
of.run(im_path=proj.filepaths[0], exclude = [scale.mask], scale=scale.measured, show_img=True, show_df=True, blur1=19, min_size=50, min_area=5000, blur2=(70,40), corr_factor=-80) 

# =============================================================================
# write skeletonize function, maybe factorize finetunings? otherwise ok
# =============================================================================

# save results
of.save(output=proj.out_dir, overwrite=True)


#%% example 2 - single objects in multiple images

for filepath, filename in zip(proj.filepaths, proj.filenames):
    # =============================================================================
    # FLOW CONTROL - continue with last index OR skip all images in out_dir
    # =============================================================================
    # if os.listdir(in_dir).index(filename) < index:
    #     continue
#    if filename in os.listdir(proj.output):
#        continue

    # =============================================================================
    # SCALE/ARENA (CURRENT) - scale.find finds drawn scale in image, 
    # poly_drawer draws arena in specified or every image
    # =============================================================================
#    if filename == "Beh_000_Standard.JPG":
#        continue
#    if any([j in filename for j in ["Standard"]]):
#        scale.current, scale.mask = scale.find(filepath, resize=0.5) 
#        proj.df.drop(index=filename, inplace=True)
#        continue
    arena = pp.polygon_maker()
    arena.draw(filepath, show=True)

    # =============================================================================
    # OBJECT FINDER - for information run help(pp.object_finder) returns 
    # dataframe (of.df) containing results, and image (of.drawn) to check
    # =============================================================================
    # initialize & run
    of = pp.standard_object_finder()
    of.run(im_path=filepath, thresholding="adaptive",mode="multi", min_size=20, min_area=2000, blur1=35, corr_factor=-10, scale=scale.current, gray_value_ref=200, exclude = [arena.mask, scale.mask], show=True) 
    
    # custom module
#    cs = pp.camera_stand_module()
#    cs.run(image=of.image, df=of.df, compare_to=of.drawn, roi_size = 500, thresholding = "otsu", show=True,  blur1=59, blur2=(29,190)) # 
#
#    # save current df to project df
#    proj.df = proj.df.reindex(columns=["project"] + list(cs.df), copy=False)
#    proj.df.update(cs.df, overwrite=True)
#    
#    # save current image 
    of.save(image=of.drawn, df=of.df, output=output_dir, overwrite=True)

    # comment out of you don't want to wait after one iterations
#    cv2.waitKey(200)


proj.save()

#%% BREAK ALL WINDOWS
cv2.destroyAllWindows()



#%% DEBUG

set(list(proj.df) + list(of.df))

cv2.namedWindow('phenopype' ,cv2.WINDOW_NORMAL)
cv2.imshow('phenopype',  of.thresh)

df = proj.df
df1 = of.df
df2 = cs.df
df1.set_index('filename', drop=True, inplace=True)

df = pd.DataFrame(data=list(zip(proj.filepaths, proj.filenames)), columns = ["filepath", "filename"])
df.index = proj.filenames
df.insert(0, "project", proj.name)
df.drop_uplicates(subset="filename", inplace=True)
df.drop(columns=["filepath"], inplace=True)
