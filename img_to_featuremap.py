# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:32:55 2022

@author: Jean-Malo
"""

import glob
import os
import cv2
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
import json


dir_path="C:\\Users\\Jean-Malo\\Documents\\Polytech\\5A\\PRD_LiDAR\\test_scripts\\data_test\\LRMdataset\\img\\"

list_img_path = glob.glob(dir_path+"*.tif")

fm_folder=dir_path+"featuremaps"

if os.path.isdir(fm_folder)==False:
    os.mkdir(fm_folder)

vgg = keras.applications.VGG16(
    include_top=False,
    weights="imagenet"
) 
backbone = Model(inputs= [vgg.layers[0].input], outputs= [vgg.layers[17].output])

i=0
for path in list_img_path:
    img=cv2.imread(path)
    feature_map=backbone.predict(np.expand_dims(img, 0))
    feature_map = feature_map.tolist() # convert to python list so it can be jsoned
    
    img_name=os.path.basename(path)
    json_name=img_name.split('.')[0]+".json"
    with open(os.path.join(fm_folder,json_name),"w") as file:
        json.dump(feature_map,file)
    i+=1
    if i%50==0:
        print("{}/{}".format(i,len(list_img_path)))
        
    """ how we can reload the json into a feature map
    with open(os.path.join(fm_folder,json_name),"r") as file:
        loaded_fm=json.load(file)
        loaded_fm=np.array(loaded_fm)
    """
    
                                 
    
    
    
    

