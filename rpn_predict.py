# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:23:08 2022

@author: Jean-Malo
"""
from tensorflow import keras
import cv2
from matplotlib import pyplot as plt
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


# script used to do predictions with model comming from rpn_train

def to_VOC_format(width, height, center_x, center_y):
    """
    Convert center coordinate format to min max coordinateformat
    """
    x_min = center_x - 0.5 * width
    y_min = center_y - 0.5 * height
    x_max = center_x + 0.5 * width
    y_max = center_y + 0.5 * height
    return x_min, y_min, x_max, y_max


def to_center_format(xmin_list, ymin_list, xmax_list, ymax_list):
    """
    Convert min max coordinate format to x_center, y_center, height and width format
    """
    height = ymax_list - ymin_list
    width = xmax_list - xmin_list
    
    center_x = xmin_list + 0.5 * width
    center_y = ymin_list + 0.5 * height
    
    return width, height, center_x, center_y


def adjust_deltas(anchor_width, anchor_height, anchor_center_x, anchor_center_y, dx, dy, dw, dh):
    """
    Adjust the anchor box with predicted offset
    """
    # ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    center_x = dx * anchor_width + anchor_center_x 
    
    # ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    center_y = dy *  anchor_height + anchor_center_y
    
    # w = np.exp(dw) * anc_width[:, np.newaxis]
    width = np.exp(dw) * anchor_width
    
    # np.exp(dh) * anc_height[:, np.newaxis]
    height = np.exp(dh) * anchor_height
    
    return width, height, center_x, center_y


def smooth_l1_loss(y_true, y_pred):
    """
    Calculates Smooth L1 loss
    """

    # Take absolute difference
    x = K.abs(y_true - y_pred)

    # Find indices of values less than 1
    mask = K.cast(K.less(x, 1.0), "float32")
    # Loss calculation for smooth l1
    loss = (mask * (0.5 * x ** 2)) + (1 - mask) * (x - 0.5)
    return loss

def custom_l1_loss(y_true, y_pred):
    """
    Regress anchor offsets(deltas) * only consider foreground boxes
    """
    offset_list= y_true[:,:,:-1]
    label_list = y_true[:,:,-1]
    
    # reshape output by the model
    y_pred = tf.reshape(y_pred, shape= (-1, 2500, 4)) #2500 is n_anchors
    
    positive_idxs = tf.where(K.equal(label_list, 1)) # select only foreground boxes
    
    # Select positive predicted bbox shifts
    bbox = tf.gather_nd(y_pred, positive_idxs)
    
    target_bbox = tf.gather_nd(offset_list, positive_idxs)
    loss = smooth_l1_loss(target_bbox, bbox)

    return K.mean(loss)

def custom_binary_loss(y_true, y_pred_objectiveness):
    '''
    Select both foreground and background class and compute cross entropy
    '''
    
    y_pred = tf.reshape(y_pred_objectiveness, shape= (-1, 2500)) #2500 is n_anchors
    y_true = tf.squeeze(y_true, -1)
    
    # Find indices of positive and negative anchors, not neutral
    indices = tf.where(K.not_equal(y_true, -1)) # ignore -1 labels

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_match_logits = tf.gather_nd(y_pred, indices)
    anchor_class = tf.gather_nd(y_true, indices)
    
    
    # Cross entropy loss
    loss = K.binary_crossentropy(target=anchor_class,
                                output=rpn_match_logits
                                )
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    
    return loss
def get_anchor_list(img_w,img_h,w_feature_map,h_feature_map):
    
    n_anchor_pos = w_feature_map * h_feature_map
    
    # width stride on the image
    x_stride = int(img_w / w_feature_map)
    y_stride = int(img_h / h_feature_map)
    
    # center (xy coordinate) of anchor location on image
    x_center = np.arange(8, img_w, x_stride) # [  0,  32,  64,  96, 128, 160, 192,...]
    y_center = np.arange(8, img_h, y_stride) # [  0,  32,  64,  96, 128, 160, 192,...]
    
    # generate all the ordered pair of x and y center
    
    # to achive this, we will use meshgrid and reshape it
    center_list = np.array(np.meshgrid(x_center, y_center,  sparse=False, indexing='xy')).T.reshape(-1,2)
    
    # visualizing the anchor positions
    """
    img_ = np.copy(self.list_img[0])
    plt.figure(figsize=(9, 6))
    for i in range(n_anchor_pos):
        cv2.circle(img_, (int(center_list[i][0]), int(center_list[i][1])), radius=1, color=(255, 0, 0), thickness=5) 
    plt.imshow(img_)
    plt.show()
    """
    # this part could be simplified because we only use one ratio and one scale
    al = []
    # aspect ratio = width/ height
    anchor_ratio_list = [1] # width = height (square)
    ratio = 1
    
    anchor_scale_list = [30] # width of each anchor box
    scale = 30
    
    # total possible anchors 
    n_anchors = n_anchor_pos * len(anchor_ratio_list) * len(anchor_scale_list)

    
    # there are total 2500 anchor centers each having 1 anchor boxes placed
    # total anchor box in the feature map will be 2500 * 1 = 2500 each anchor box is denoted by 4 numbers.
    anchor_list = np.zeros(shape= (n_anchors, 4))
    
    count = 0
    # to get height and width given ratio and scale, we will use formula given above
    # for each anchor location
    for center in center_list:
        center_x, center_y = center[0], center[1]
        """
        # for each ratio
        for ratio in anchor_ratio_list:
            # for each scale
            for scale in anchor_scale_list:
        """
        # compute height and width and scale them by constant factor
        h = pow(pow(scale, 2)/ ratio, 0.5) # h = scale
        w = h * ratio # w = h

        # as h and w would be really small, we will scale them with some constant (in our case, stride width and height)
        #h *= x_stride
        #w *= y_stride


        # * at this point we have height and width of anchor and centers of anchor locations
        # putting anchor 9 boxes at each anchor locations
        anchor_xmin = center_x - 0.5 * w
        anchor_ymin = center_y - 0.5 * h
        anchor_xmax = center_x + 0.5 * w
        anchor_ymax = center_y + 0.5 * h
        al.append([center_x, center_y, w, h])
        # append the anchor box to anchor list
        anchor_list[count] = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]
        count += 1
    #print("anchor list shape",self.anchor_list.shape)
    
    """
    h = w = 800
    # select anchor boxes which are inside the image
    inside_anchor_idx_list = np.where(
        (anchor_list[:,0] >= 0) &
        (anchor_list[:,1] >= 0) &
        (anchor_list[:,2] <= w) &
        (anchor_list[:,3] <= h))[0]
    #print(self.inside_anchor_idx_list.shape)
    inside_anchor_list = anchor_list[inside_anchor_idx_list]
    #n_inside_anchor = len(self.inside_anchor_idx_list)
    
    #print("number of anchor \"inside\" the image", n_inside_anchor)
    """
    return anchor_list

folder_path="C:\\Users\\Jean-Malo\\Documents\\Polytech\\5A\\PRD_LiDAR\\test_scripts\\data_test\\LRMdataset\\img"

json_path = os.path.join(folder_path,"bbox")


img_name='LRM_tot_clem_120.tif'
model_name="rpn_120_nice_charb_all.h5"
bbox_conf_threshold=0.96

json_name=img_name.split('.')[0]+".json"

# Load the image 

img = cv2.imread(os.path.join(folder_path, img_name))

print("img shape",img.shape)
img_h, img_w,_ = img.shape


plt.imshow(img,cmap="gray")
plt.show()


# Load bbox ground truth 


with open(os.path.join(json_path,json_name),'r') as file:
    bbox_list = json.load(file)
print("bbox_list",bbox_list)
# visualize ground truth box
img_ = np.copy(img)
for i, bbox in enumerate(bbox_list):
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)  

plt.imshow(img_,cmap='gray') 
plt.show() 


# Backbone network 

vgg = keras.applications.VGG16(
    include_top=False,
    weights="imagenet"
) 

backbone = Model(inputs= [vgg.layers[0].input], outputs= [vgg.layers[17].output])

# examining the shape of feature map
feature_maps = backbone.predict(np.expand_dims(img, 0))
_, w_feature_map, h_feature_map, _ = feature_maps.shape
# number of posssible anchor positions
n_anchor_pos = w_feature_map * h_feature_map
print("feature_maps.shape",feature_maps.shape)
#Our feature map dimension size is 50*50 (512 depth) = 2500, so total anchor centers would be 2500
n_anchors=2500


# load models
RPN = keras.models.load_model(model_name, compile=False)

# get the offset and objectiveness score
anchor_deltas, objectiveness_score = RPN.predict(feature_maps)


# shape both predictions
anchor_deltas = anchor_deltas.reshape(-1, n_anchors, 4) # 50*50, 4
objectiveness_score = objectiveness_score.reshape(-1, n_anchors) # 50*50*9, 1

# parse anchor deltas
dx = anchor_deltas[:, :, 0]
dy = anchor_deltas[:, :, 1]
dw = anchor_deltas[:, :, 2]
dh = anchor_deltas[:, :, 3]

print("shape of predicted anchor delta and objectiveness",anchor_deltas.shape, objectiveness_score.shape)

#%% Filtering

anchor_list=get_anchor_list(800,800,50,50) #800 50
anchor_list = anchor_list.squeeze()
anchor_list = np.expand_dims(anchor_list, 0)
# for each anchor box, convert coordinate format (min_x, min_y, max_x, max_y to height, width, center_x)
anchor_width, anchor_height, anchor_center_x, anchor_center_y =  to_center_format(
    anchor_list[0][:, 0], 
    anchor_list[0][:, 1],
    anchor_list[0][:, 2],
    anchor_list[0][:, 3])

# get the region proposals (adjust the anchor boxes to the offset predicted by our model)
roi_width, roi_height, roi_center_x, roi_center_y = adjust_deltas(anchor_width,
                                                      anchor_height,
                                                      anchor_center_x,
                                                      anchor_center_y,
                                                      dx,
                                                      dy,
                                                      dw,
                                                      dh)
# ROI format conversion (width, height,center x, centery ===> min x, min y, max x, max y)
roi_min_x, roi_min_y, roi_max_x, roi_max_y = to_VOC_format(roi_width, roi_height, roi_center_x, roi_center_y)
roi = np.vstack((roi_min_x, roi_min_y, roi_max_x, roi_max_y)).T
# clipping the predicted boxes to the image
roi = np.clip(roi, 0, 800)
# remove predicted boxes with either height or width < threshold.
min_size = 16
width = roi[:, 2] - roi[:, 0] # xmax - xmin
height = roi[:, 3] - roi[:, 1] # ymin - ymax
keep = np.where((width > min_size) & (height> min_size))[0]
roi = roi[keep]
score = objectiveness_score[:,keep]
# Sort all (proposal, score) pairs by score from highest to lowest.
sorted_idx = score.flatten().argsort()[::-1]
score_sorted = score[:, sorted_idx]
roi_sorted = roi[sorted_idx]
print("shape of sorted idx, score and roi",sorted_idx.shape, score_sorted.shape, roi_sorted.shape)

#%% Non max suppression
# select top N proposals (top 12000)
pre_NMS_topN = 12000 #12000
score_sorted = score_sorted[:, :pre_NMS_topN]
roi_sorted = roi_sorted[:pre_NMS_topN]
#TO DO

#%% Visualizing top anchor boxes
img_ = np.copy(img)
for i, bbox in enumerate(bbox_list):
    cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)

#for i in range(0, 6):  # 5625// 2
i=0
while score_sorted[0,i]>bbox_conf_threshold:
    x_min = int(roi_sorted[i][0])
    y_min = int(roi_sorted[i][1])
    x_max = int(roi_sorted[i][2])
    y_max = int(roi_sorted[i][3])
    cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2) 
    cv2.putText(img_,
                "{:.2f}".format(score_sorted[0,i]),
                (x_min-10,y_min),
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                1, #font size
                (255, 0, 0), #font color
                2) #stroke
    print("confidence for",i,"th bbox :",score_sorted[0,i])
    i+=1
    plt.imshow(img_)
    
print("next bboxe conf values")
stop = i + 5
while i < stop:
    
    print("confidence for",i,"th bbox :",score_sorted[0,i])
    i+=1

    
plt.imshow(img_)
plt.show()

#pistes : verifier que le shuffle fonctionne bien + passe de 800x800 a 4 400x400