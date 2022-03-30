# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:45:42 2022

@author: Jean-Malo
"""
import cv2
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
import json
import os

# script that use a training image (lidar image + bbox json file) to see if the feature maps returned by VGG contain high activation value on our objects

# path to an image directory
dir_path="data\\train\\"
img_name="LRM_tot_clem_184.tif"
json_name=img_name.replace(".tif",".json")

img_path=os.path.join(dir_path, img_name)
fm_path=dir_path+"\\featuremaps\\"+json_name
bbox_path=dir_path+"\\bbox\\"+json_name


img = cv2.imread(img_path)
plt.imshow(img)
plt.show()

with open(bbox_path,'r') as file:
    bboxes=json.load(file)
    
bboxes=np.array(bboxes)
reduce_bboxes=bboxes//16 # because 800/16 = 50
reduce_centers=[]
for b in reduce_bboxes:
    reduce_centers.append([ (b[0]+b[2])//2, (b[1]+b[3])//2 ])
# each bbox is [mincolumn,minraw,maxcolumn,maxraw]
print(bboxes)
print(reduce_bboxes)
print(reduce_centers)

img_ = np.copy(img)
k=1
for bbox in bboxes:
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,255,0), thickness=3)  
    cv2.putText(img_,
                str(k),
                (bbox[0],bbox[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                1, #font size
                (0, 255, 0), #font color
                2) #stroke
    k+=1
plt.imshow(img_.astype(int), cmap="gray")
plt.show()

""" # if you want to compute the fm (and not load them form json file)
vgg = keras.applications.VGG16(
    include_top=False,
    weights="imagenet"
) 
backbone = Model(inputs= [vgg.layers[0].input], outputs= [vgg.layers[17].output])
fm=backbone.predict(np.expand_dims(img, 0))
#"""

#""" # if you just want to load the featuremap
with open(fm_path,'r') as file:
    fm=json.load(file)
fm=np.array(fm)
#"""

print("fm shape", fm.shape)

""" # I used this part of the script to try to find wich channels were always activ on our charcoal kilns
l=[18,44,172,221,248,278,357,402,419,454,459,495,502]
s=[18]
activ_on_104=[4, 15, 33, 34, 36, 44, 45, 55, 58, 64, 76, 84, 89, 94, 98, 103, 109, 117, 118, 124, 149, 155, 160, 163, 172, 174, 179, 185, 186, 188, 194, 199, 201, 207, 223, 227, 240, 248, 258, 259, 261, 272, 284, 298, 306, 315, 319, 325, 335, 341, 343, 344, 345, 351, 356, 357, 367, 369, 373, 381, 387, 400, 409, 410, 414, 416, 417, 430, 431, 441, 450, 452, 454, 458, 459, 462, 463, 471, 486, 488, 491, 495, 496, 502, 504, 505, 506, 510]
activ_on_105=[4, 15, 18, 20, 33, 36, 42, 44, 45, 50, 55, 56, 71, 76, 84, 88, 94, 98, 99, 103, 108, 109, 117, 118, 139, 149, 152, 155, 160, 163, 166, 171, 172, 174, 179, 185, 194, 199, 201, 207, 227, 248, 250, 254, 259, 261, 275, 284, 298, 306, 315, 319, 325, 335, 341, 343, 345, 356, 357, 359, 369, 373, 381, 387, 400, 404, 407, 411, 414, 416, 417, 419, 430, 441, 450, 452, 454, 458, 459, 463, 464, 486, 488, 491, 495, 499, 502, 504, 505, 510]
activ_on_56=[12, 15, 21, 36, 40, 42, 49, 50, 72, 88, 89, 123, 152, 155, 172, 186, 190, 199, 246, 250, 254, 261, 276, 283, 296, 300, 306, 315, 319, 323, 328, 345, 359, 368, 369, 372, 381, 396, 414, 438, 441, 458, 483, 494, 502, 504, 507]
activ_mean_104=[4, 33, 36, 44, 45, 55, 64, 76, 84, 94, 98, 103, 109, 117, 118, 155, 160, 163, 172, 174, 179, 185, 194, 199, 201, 207, 227, 248, 258, 259, 261, 272, 284, 298, 306, 325, 335, 341, 344, 345, 351, 357, 373, 387, 400, 410, 417, 430, 431, 441, 450, 452, 454, 459, 462, 471, 488, 491, 495, 504, 505, 506, 510]
activ_mean_105=[4, 18, 33, 36, 44, 45, 55, 56, 76, 84, 88, 94, 98, 103, 109, 117, 139, 155, 160, 163, 166, 171, 172, 174, 179, 185, 194, 199, 201, 207, 227, 248, 250, 259, 261, 275, 284, 298, 306, 325, 335, 345, 356, 357, 373, 381, 387, 400, 404, 411, 414, 416, 417, 419, 441, 450, 452, 454, 458, 459, 486, 488, 491, 495, 499, 504, 505, 510]
activ_mean_both=[4, 33, 36, 44, 45, 55, 76, 84, 94, 98, 103, 109, 117, 155, 160, 163, 172, 174, 179, 185, 194, 199, 201, 207, 227, 248, 259, 261, 284, 298, 306, 325, 335, 345, 357, 373, 387, 400, 417, 441, 450, 452, 454, 459, 488, 491, 495, 504, 505, 510]

count=0
temp=[]
for a in activ_on_56:
    if (a in activ_mean_both) == True:
        count+=1
        temp.append(a)
print(count,temp)
#"""

count=0
activ_channels=[]
value=[]

for i in range(512):
    if np.max(fm[0,:,:,i])==0:
        count+=1
    
    print("\nimg",i)
    img_ = np.copy(fm[0,:,:,i])
    #print(img_.shape)
    print("max activation",np.max(img_))
    
    activ_on_charb=False
    for j in range(len(reduce_centers)):
        #print(reduce_centers[j])
        activation=img_[reduce_centers[j][1],reduce_centers[j][0]]
        if activation > 0:#np.mean(img_):
            activ_on_charb=True
            print("center charb",j+1,"activation=",activation)
    
    for j in range(len(reduce_bboxes)):
        b=reduce_bboxes[j]
        activation=np.mean(img_[b[1]:b[3]+1,b[0]:b[2]+1])
        if activation > 0:#np.mean(img_):
            activ_on_charb=True
            value.append(activation)
            print("charb",j+1,"mean activation=",activation)
            #print(img_[b[1]:b[3]+1,b[0]:b[2]+1].size)
    
    if activ_on_charb==True:
        activ_channels.append(i)
        
        #"""
        plt.subplot(131)
        plt.title("featuremap "+str(i))
        plt.imshow(img_,interpolation=None)
    
        
        for c in reduce_centers:
            img_[c[1],c[0]]=np.max(img_)
        plt.subplot(132)
        plt.title("fm + center")
        plt.imshow(img_,interpolation=None)
        
        
        for bbox in reduce_bboxes:
            cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(np.max(img_)), thickness=-1)  
        plt.subplot(133)
        plt.title("fm + bbox")
        plt.imshow(img_,interpolation=None,)
        plt.show()
        
        #"""
        a=input("press enter for next image (exit to stop) \n")
        if a=="exit":
            break
        #"""
    else:
        print("no activation on any charb")
print("\n")
print(count,"channels without any activation (filled with 0)")
print("channels with activation on at list one of the charb:",activ_channels)
print("len:",len(activ_channels))
#print("activation values:",value)
plt.title("activation value repartition")
plt.hist(value)
plt.show()
#"""

