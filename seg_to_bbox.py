# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:13:04 2022

@author: Jean-Malo
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import skimage.io as io
import json


from skimage.measure import label, regionprops

# script that transform segmentation image (object as 0, background as 1) into bbox list json file


seg_folder_path="C:\\Users\\Jean-Malo\\Documents\\Polytech\\5A\\PRD_LiDAR\\test_scripts\\data_test\\LRMdataset\\nice_C1_val"


# read all images (check for tif and png in the folder) 
list_seg_path=glob.glob(seg_folder_path+"\\*.tif")+glob.glob(seg_folder_path+"\\*.png")

json_dir=os.path.join(seg_folder_path,"bbox")
if os.path.isdir(json_dir)==False:
    os.mkdir(json_dir)

img_size=None
for path in list_seg_path:
    img = io.imread(path, as_gray=True)
    
    # so it works when the image is coded 0, 1 or 0, 255
    if np.max(img)!=1:
        img=img/np.max(img)
        
    if img_size is None:
        img_size = img.shape[0]
        
    img = np.abs(img-int(np.max(img))) # because 1 should be object and 0 background
    label_img = label(img)
    regions = regionprops(label_img)
    
    bbox_list=[]

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        # the bbox attribute returns [min,max[ interval for row and column containing component's pixels
        # and we want a 1 pixel border of background
        minc -= 2
        minr -= 2
        maxr += 1
        maxc += 1
        
        if minc < 0:
            minc=0
        if minr < 0:
            minr=0
        if maxr > img_size-1:
            maxr=img_size-1
        if maxc > img_size-1:
            maxc=img_size-1
        
        # we don't save the bbox that are too small
        if maxc-minc > 10 and maxr-minr >10 :#and maxc-minc<100 and maxr-minr<100:
            bbox_list.append([minc,minr,maxc,maxr])
           
        # if you want to plot the img with found bbox
        # img[(minr, minr, maxr, maxr, minr),(minc, maxc, maxc, minc, minc)] = np.max(img)
            
    
    img_name=os.path.basename(path)
    json_name=img_name.replace(".tif",".json")
    json_name=json_name.replace(".png",".json")
    with open(os.path.join(json_dir,json_name),"w") as file:
        json.dump(bbox_list,file)
        
