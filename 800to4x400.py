# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:53:18 2022

@author: Jean-Malo
"""

import glob
import os
import skimage.io as io
import skimage

# change 800x800 images into 4 400x400 images
# 4 images lightly overlap on the center of the original image
# because in our dataset there is always an object on the center of each image

# path where there should be a folder with images and a folder with segmentations
folder_path="C:\\Users\\Jean-Malo\\Documents\\Polytech\\5A\\PRD_LiDAR\\test_scripts\\data_test\\LRMdataset\\rpn_400"

list_seg_path = glob.glob(folder_path+"\\C1\\*.tif")

new_seg_folder=folder_path+"\\new_C1"

if os.path.isdir(new_seg_folder)==False:
    os.mkdir(new_seg_folder)

list_img_path = glob.glob(folder_path+"\\img\\*.tif")

new_img_folder = folder_path+"\\new_img"

if os.path.isdir(new_img_folder)==False:
    os.mkdir(new_img_folder)

for path in list_seg_path:
    #print(path)
    img_name=os.path.basename(path).split('.')[0]
    img=io.imread(path)
    print(img.shape)
    io.imsave(new_seg_folder+"\\"+img_name+"_0.tif",skimage.img_as_ubyte(img[50:450,50:450]),check_contrast=False)
    io.imsave(new_seg_folder+"\\"+img_name+"_1.tif",skimage.img_as_ubyte(img[50:450,350:750]),check_contrast=False)
    io.imsave(new_seg_folder+"\\"+img_name+"_2.tif",skimage.img_as_ubyte(img[350:750,50:450]),check_contrast=False)
    io.imsave(new_seg_folder+"\\"+img_name+"_3.tif",skimage.img_as_ubyte(img[350:750,350:750]),check_contrast=False)
    
for path in list_img_path:
    #print(path)
    img_name=os.path.basename(path).split('.')[0]
    img=io.imread(path)
    print(img.shape)
    io.imsave(new_img_folder+"\\"+img_name+"_0.tif",skimage.img_as_ubyte(img[50:450,50:450]),check_contrast=False)
    io.imsave(new_img_folder+"\\"+img_name+"_1.tif",skimage.img_as_ubyte(img[50:450,350:750]),check_contrast=False)
    io.imsave(new_img_folder+"\\"+img_name+"_2.tif",skimage.img_as_ubyte(img[350:750,50:450]),check_contrast=False)
    io.imsave(new_img_folder+"\\"+img_name+"_3.tif",skimage.img_as_ubyte(img[350:750,350:750]),check_contrast=False)
    

