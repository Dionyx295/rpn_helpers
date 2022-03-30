# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:28:35 2022

@author: Jean-Malo
"""
import os
import cv2
from matplotlib import pyplot as plt
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model

# script used to analyse the feature maps inside the RPN

dir_path="data\\train"
img_name="LRM_tot_clem_202.tif"
model_name="model\\rpn_model.h5"
json_name=img_name.replace(".tif",".json")

img_path=os.path.join(dir_path, img_name)
fm_path=dir_path+"\\featuremaps\\"+json_name
bbox_path=dir_path+"\\bbox\\"+json_name
bbox_conf_threshold=0.8

json_name=img_name.split('.')[0]+".json"

# Load the image 

img = cv2.imread(os.path.join(dir_path, img_name))

print("img shape",img.shape)
img_h, img_w,_ = img.shape


plt.imshow(img,cmap="gray")
plt.show()


# Load bbox ground truth 


with open(bbox_path,'r') as file:
    bbox_list = json.load(file)
print("bbox_list",bbox_list)
# visualize ground truth box
img_bbox = np.copy(img)
for i, bbox in enumerate(bbox_list):
    cv2.rectangle(img_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)  

plt.imshow(img_bbox,cmap='gray') 
plt.show() 


# Backbone network 
""" # if you don't want to use pre computed featuremap
vgg = keras.applications.VGG16(
    include_top=False,
    weights="imagenet"
) 

backbone = Model(inputs= [vgg.layers[0].input], outputs= [vgg.layers[17].output])

feature_maps = backbone.predict(np.expand_dims(img, 0))
"""

with open(fm_path,'r') as file:
    feature_maps=json.load(file)
feature_maps=np.array(feature_maps)

# load models
RPN = keras.models.load_model(model_name, compile=False)

# get the offset and objectiveness score
anchor_deltas, objectiveness_score = RPN.predict(feature_maps)

RPN.summary()
first_conv=RPN.get_layer("conv1")
obj_conv=RPN.get_layer("objectivess_score")


# featuremaps after the first convolution layer
x=first_conv(feature_maps)
print(x.shape)

# define how you want to browse those featuremaps
for i in range(3):
    print("img", i)
    #plt.subplot(121)
    img_=np.copy(x[0,:,:,i])
    plt.imshow(img_)
    """
    plt.subplot(122)
    plt.imshow(img_bbox)
    """
    plt.show()
    """
    a=input("press enter for next image (exit to stop) \n")
    if a=="exit":
        break
    """
    
# featuremaps after the objectiveness layer 
# can be seen as heat map of the prédiction
x=obj_conv(x)
print(x.shape)

img_=np.copy(x[0,:,:,0])
plt.imshow(img_)
plt.show()

print("max activation :",np.max(img_))

# i didn't show the featuremaps from the delta layer because they can't be represented as images as the predictions correspond to set of values
