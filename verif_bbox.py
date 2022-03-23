# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:06:50 2022

@author: Jean-Malo
"""
import glob
import time
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage import io

# script to check if the bounding boxes match the segmentation images


#path to lidar images 
# the img directory should contain a bbox directory (one json file per image)
dir_path="C:\\Users\\Jean-Malo\\Documents\\Polytech\\5A\\PRD_LiDAR\\test_scripts\\data_test\\LRMdataset\\nice_charb\\"

#path to segmentation
dir_C1="C:\\Users\\Jean-Malo\\Documents\\Polytech\\5A\\PRD_LiDAR\\test_scripts\\data_test\\LRMdataset\\nice_C1\\"

list_img_path = glob.glob(dir_path+"*.tif")
list_json_path = glob.glob(dir_path+"bbox\\*.json")
list_C1_path = glob.glob(dir_C1+"*.png") # beware of the format

t=time.perf_counter()
list_img=[]
for path in list_img_path:
    list_img.append(io.imread(path))
print("len list img",len(list_img),time.perf_counter()-t,"s")

t=time.perf_counter()
list_C1=[]
for path in list_C1_path:
    list_C1.append(cv2.imread(path))
print("len list C1",len(list_C1),time.perf_counter()-t,"s")

t=time.perf_counter()
list_bboxes=[]
for path in list_json_path:
    with open(path,'r') as file:
        list_bboxes.append(json.load(file))

print("len bboxes list",len(list_bboxes),time.perf_counter()-t,"s")

t=time.perf_counter()
i=0
for img in list_img:
    if i>=0:
        img_ = np.copy(img)
        
        """
        # for MNT images (32bits) some modifications need to be done
        img_ = img_-np.min(img_)
        img_= img_/np.max(img_)*255
        #img_ = int
        print(img_.shape)
        img_=cv2.cvtColor(img_,cv2.COLOR_GRAY2BGR)
        print(img_.shape)
        #"""
        
        for bbox in list_bboxes[i]:
            cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,255,0), thickness=3)  
        
        plt.subplot(121)
        plt.imshow(img_.astype(int), cmap="gray")
        plt.subplot(122)
        plt.imshow(list_C1[i])
        plt.show()
        print(os.path.basename(list_img_path[i]))
        print(os.path.basename(list_C1_path[i]))
        
        
        a=input("press enter for next image  \n")
        if a=="exit":
            break
    i+=1
