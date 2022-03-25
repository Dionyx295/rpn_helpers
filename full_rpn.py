# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:24:45 2022

@author: Jean-Malo
"""
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
#from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import initializers
import os
import skimage.io as io

# script that take one image and one bbox json file as input,
# plot the result of training (Region Proposal Network) on this same image

# based on https://github.com/martian1231/regionProposalNetworkInFasterRCNN/blob/master/region_proposal_network_in_keras_from_scratch.ipynb

#%% Functions def

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

def compute_deltas(base_center_x, base_center_y, base_width, base_height, inside_anchor_width, inside_anchor_height, inside_anchor_center_x, inside_anchor_center_y):
    """
    computing offset of achor box to the groud truth box
    """
    dx = (base_center_x - inside_anchor_center_x)/ inside_anchor_width  # difference in centers of ground truth and anchor box across x axis
    dy = (base_center_y - inside_anchor_center_y)/  inside_anchor_height  # difference in centers of ground truth and anchor box across y axis
    dw = np.log(base_width/ inside_anchor_width) # log on ratio between ground truth width and anchor box width
    dh = np.log(base_height/ inside_anchor_height) # log on ratio between ground truth height and anchor box height
    return dx, dy, dw, dh

def IOU(box1, box2):
    """
    Compute overlap (IOU) between box1 and box2
    """
    
    # ------calculate coordinate of overlapping region------
    # take max of x1 and y1 out of both boxes

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    
    # take min of x2 and y2 out of both boxes
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # check if they atleast overlap a little
    if (x1 <= x2 and y1 <= y2):
        # ------area of overlapping region------
        width_overlap = (x2 - x1)
        height_overlap = (y2 - y1)
        area_overlap = width_overlap * height_overlap
    else:
        return 0
    
    # ------computing union------
    # sum of area of both the boxes - area_overlap
    
    # height and width of both boxes
    width_box1 = (box1[2] - box1[0])
    height_box1 = (box1[3] - box1[1])
    
    width_box2 = (box2[2] - box2[0])
    height_box2 = (box2[3] - box2[1])
    
    # area of box1 and box2
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # union (including 2 * overlap area (double count))
    area_union_overlap = area_box1 + area_box2
    
    # union
    area_union = area_union_overlap - area_overlap
    
    # compute IOU
    iou = area_overlap/ area_union
    
    return iou

def sample_anchors_pre(df, n_samples= 1024, neg_ratio= 0.5):
    """
    Sample total of n samples across both the class (background and foreground),
    If one of the class have less samples than n/2, we will sample from majority class to make up for short.
    """
    n_foreground = int((1-neg_ratio) * n_samples)
    n_backgroud = int(neg_ratio * n_samples)
    foreground_index_list = df[df.label == 1].index.values
    background_index_list = df[df.label == 0].index.values

    # check if we have excessive positive samples
    if len(foreground_index_list) > n_foreground:
        # mark excessive samples as -1 (ignore)
        ignore_index = foreground_index_list[n_backgroud:]
        df.loc[ignore_index, "label"] = -1

    # sample background examples if we don't have enough positive examples to match the anchor batch size
    if len(foreground_index_list) < n_foreground:
        diff = n_foreground - len(foreground_index_list)
        # add remaining to background examples
        n_backgroud += diff

    # check if we have excessive background samples
    if len(background_index_list) > n_backgroud:
        # mark excessive samples as -1 (ignore)
        ignore_index = background_index_list[n_backgroud:]
        df.loc[ignore_index, "label"] = -1

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
    y_pred = tf.reshape(y_pred, shape= (-1, n_anchors, 4))
    
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
    
    y_pred = tf.reshape(y_pred_objectiveness, shape= (-1, n_anchors))
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

#%%main    
if __name__ == '__main__':
    folder_path="./data/train" #"C:\\Users\\Jean-Malo\\Documents\\Polytech\\5A\\PRD_LiDAR\\test_scripts\\data_test\\LRMdataset\\img"
    json_path = os.path.join(folder_path,"bbox")
    
    #%% Load the image 
    
    #img_name='mnt_solidar_gapclosed.sdat_101.tif'
    img_name="LRM_tot_clem_202.tif"
    json_name=img_name.replace('.tif',".json")
    img = io.imread(os.path.join(folder_path, img_name))
    img_for_plt = np.copy(img)
    
    """ # if you use 32 bits images
    img_for_plt = img_for_plt-np.min(img_for_plt)
    img_for_plt= img_for_plt/np.max(img_for_plt)*255
    """
    img_for_plt=cv2.cvtColor(img_for_plt,cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) #because vgg need 3 channels

    
    print("img shape",img.shape)
    img_h, img_w,_ = img.shape
    
    

    plt.imshow(img_for_plt.astype(int))
    plt.show()
    
    
    #%% Load bbox ground truth 
    
    
    with open(os.path.join(json_path,json_name),'r') as file:
      bbox_list = json.load(file)
    print("bbox_list",bbox_list)
    # visualize ground truth box
    img_ = np.copy(img_for_plt)
    for i, bbox in enumerate(bbox_list):
        cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)  
        
    plt.imshow(img_.astype(int)) 
    plt.show() 
    
    
    #%% Backbone network 
    
    vgg = keras.applications.VGG16(
        include_top=False,
        weights="imagenet"
    ) 
    
    backbone = Model(inputs= [vgg.layers[0].input], outputs= [vgg.layers[17].output])
    
    # img_ = np.copy(img)[np.newaxis,:]
    # print(backbone.predict(img_).shape)
    
    # examining the shape of feature map
    feature_maps = backbone.predict(np.expand_dims(img, 0))
    _, w_feature_map, h_feature_map, _ = feature_maps.shape
    # number of posssible anchor positions
    n_anchor_pos = w_feature_map * h_feature_map
    print("feature_maps.shape",feature_maps.shape)
    #Our feature map dimension size is 50*50 (512 depth) = 2500, so total anchor centers would be 2500
    
    
    #%% Creating anchor 
    
    
    # width stride on the image
    x_stride = int(img_w / w_feature_map)
    y_stride = int(img_h / h_feature_map)
    
    # center (xy coordinate) of anchor location on image
    x_center = np.arange(8, img_w, x_stride) # [  0,  32,  64,  96, 128, 160, 192,...]
    y_center = np.arange(8, img_h, y_stride) # [  0,  32,  64,  96, 128, 160, 192,...]
    
    # generate all the ordered pair of x and y center
    
    # to achive this, we will use meshgrid and reshape it
    center_list = np.array(np.meshgrid(x_center, y_center,  sparse=False, indexing='ij')).T.reshape(-1,2)
    
    
    # visualizing the anchor positions
    img_ = np.copy(img_for_plt)
    plt.figure(figsize=(9, 6))
    for i in range(n_anchor_pos):
        cv2.circle(img_, (int(center_list[i][0]), int(center_list[i][1])), radius=1, color=(255, 0, 0), thickness=5) 
    plt.imshow(img_.astype(int))
    plt.show()
    
    # this part could be simplified because we only use one ratio and one scale
    al = []
    # aspect ratio = width/ height
    anchor_ratio_list = [1] # width = height (square)
    anchor_scale_list = [30] # width of each anchor box
    
    # total possible anchors 
    n_anchors = n_anchor_pos * len(anchor_ratio_list) * len(anchor_scale_list)
    
    # number of object in the image
    n_object = len(bbox_list)
    
    # there are total 2500 anchor centers each having 9 anchor boxes placed
    # total anchor box in the feature map will be 2500 * 9 = 22500 each anchor box is denoted by 4 numbers.
    anchor_list = np.zeros(shape= (n_anchors, 4))
    
    count = 0
    # to get height and width given ratio and scale, we will use formula given above
    # for each anchor location
    for center in center_list:
        center_x, center_y = center[0], center[1]
        # for each ratio
        for ratio in anchor_ratio_list:
            # for each scale
            for scale in anchor_scale_list:
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
    print("anchor list shape",anchor_list.shape)
    
    # visualize some anchor boxes 
    img_ = np.copy(img_for_plt)
    # mid anchor center = 2500/2 = 1250
    for i in range(len(anchor_list)//2+20,len(anchor_list)//2+25):  # 1250 * 2 = 2500 (2 anchors corresponds to mid anchor center)
        x_min = int(anchor_list[i][0])
        y_min = int(anchor_list[i][1])
        x_max = int(anchor_list[i][2])
        y_max = int(anchor_list[i][3])
        cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 
    
    for i, bbox in enumerate(bbox_list):
        cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)
        
    plt.imshow(img_.astype(int))
    plt.show()
    
    
    h = img_h
    w = img_w
    # select anchor boxes which are inside the image
    inside_anchor_idx_list = np.where(
        (anchor_list[:,0] >= 0) &
        (anchor_list[:,1] >= 0) &
        (anchor_list[:,2] <= w) &
        (anchor_list[:,3] <= h))[0]
    print(inside_anchor_idx_list.shape)
    inside_anchor_list = anchor_list[inside_anchor_idx_list]
    n_inside_anchor = len(inside_anchor_idx_list)
    
    print("number of anchor \"inside\" the image",n_inside_anchor)
    
    
    #%% Computing IOU for each anchor boxes with each ground truth box 
    
    
    iou_list = np.zeros((n_inside_anchor, n_object))
    print(iou_list.shape)
    # for each ground truth box
    for gt_idx, gt_box in enumerate(bbox_list):
        # for each anchor boxes
        for anchor_idx, anchor_box in enumerate(inside_anchor_list):
            # compute IOU
            iou_list[anchor_idx][gt_idx] = IOU(gt_box, anchor_box)
    # convert to dataframe
    
    # add anchor_id
    data = {"anchor_id" :inside_anchor_idx_list}
    
    # add object column and corresponding IOU
    data.update({f"object_{idx}_iou":iou_list[:, idx] for idx in range(n_object)})
    
    # for each anchor box assign max IOU among all objects in the image
    data["max_iou"] = iou_list.max(axis= 1)
    
    # for each anchorbox assign ground truth having maximum IOU
    data["best_gt"] = iou_list.argmax(axis= 1)
    
    df_iou = pd.DataFrame(data)
    
    df_iou.info(verbose=True)
    
    
    #%% Remarkable values 
    
    
    # getting anchor boxes having maximum IOU for each ground truth boxes
    best_ious = df_iou.drop(["anchor_id", "max_iou", "best_gt"],axis= 1).max().values
    print(f"Top IOUs for each object in the image: {best_ious}")
    
    # getting anchor box idx having maximum overlap with ground truth boxes * ignoring anchor id column
    best_anchors = df_iou.drop(["anchor_id", "max_iou", "best_gt"],axis= 1).values.argmax(axis= 0)
    print(f"Top anchor boxes index: {best_anchors}")
    
    # get all the anchor boxes having same IOU score
    top_anchors = np.where(iou_list == best_ious)[0]
    print(f"Anchor boxes with same IOU score: {top_anchors}")
    
    # visualizing top anchor boxes
    img_ = np.copy(img_for_plt)
    for i in top_anchors:  # 5625// 2
        x_min = int(inside_anchor_list[i][0])
        y_min = int(inside_anchor_list[i][1])
        x_max = int(inside_anchor_list[i][2])
        y_max = int(inside_anchor_list[i][3])
        cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 
    
    for i, bbox in enumerate(bbox_list):
        cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)
    
    plt.imshow(img_.astype(int))
    plt.show()
    
    
    #%% Labeling each anchor box based on IOU threshold 
    
    # create dummy column
    label_column = np.zeros(df_iou.shape[0], dtype= np.int32)
    label_column.fill(-1)
    
    # label top anchor boxes as 1 # contains object
    label_column[top_anchors] = 1
    
    # label anchor boxes having IOU > 0.4 with ground truth boxes as 1 # contains object
    label_column[np.where(df_iou.max_iou.values >= 0.2)[0]] = 1
    
    # label anchor boxes having IOU < 0.1 with ground truth boxes as 0 # background
    label_column[np.where(df_iou.max_iou.values < 0.01)[0]] = 0
    
    # add column to the iou dataframe
    df_iou["label"] = label_column
    
    print("qty of foreground anchor",len(df_iou[df_iou.label == 1].index.values))
    print("qty of background anchor",len(df_iou[df_iou.label == 0].index.values))
    sample_anchors_pre(df_iou)
    print("qty of foreground anchor after sample",len(df_iou[df_iou.label == 1].index.values))
    print("qty of background anchor after sample",len(df_iou[df_iou.label == 0].index.values))
    
    
    #%% Computing offset
    
    # for each valid anchor box coordinate, convert coordinate format
    inside_anchor_width, inside_anchor_height, inside_anchor_center_x, inside_anchor_center_y =  to_center_format(
        inside_anchor_list[:, 0],
        inside_anchor_list[:, 1],
        inside_anchor_list[:, 2],
        inside_anchor_list[:, 3])
    
    
    # for each ground truth box corresponds to each anchor box coordinate, convert coordinate format
    gt_coordinates = []
    for idx in df_iou.best_gt:
        gt_coordinates.append(bbox_list[idx])
    gt_coordinates = np.array(gt_coordinates)
    
    base_width, base_height, base_center_x, base_center_y =  to_center_format(
        gt_coordinates[:, 0], 
        gt_coordinates[:, 1],
        gt_coordinates[:, 2],
        gt_coordinates[:, 3])
    
    # the code below prevents from "exp overflow"
    eps = np.finfo(inside_anchor_width.dtype).eps
    inside_anchor_height = np.maximum(inside_anchor_height, eps)
    inside_anchor_width = np.maximum(inside_anchor_width, eps)
    
    # computing offset given by above expression
    dx = (base_center_x - inside_anchor_center_x)/ inside_anchor_width  # difference in centers of ground truth and anchor box across x axis
    dy = (base_center_y - inside_anchor_center_y)/  inside_anchor_height  # difference in centers of ground truth and anchor box across y axis
    dw = np.log(base_width/ inside_anchor_width) # log on ratio between ground truth width and anchor box width
    dh = np.log(base_height/ inside_anchor_height) # log on ratio between ground truth height and anchor box height
    
    # adding offsets to the df
    df_iou["dx"] = dx
    df_iou["dy"] = dy
    df_iou["dw"] = dw
    df_iou["dh"] = dh
    
    # labels for all possible anchors
    label_list = np.empty(n_anchors, dtype = np.float32)
    label_list.fill(-1)
    label_list[df_iou.anchor_id.values] = df_iou.label.values
    label_list = np.expand_dims(label_list, 0)
    label_list = np.expand_dims(label_list, -1)
    
    df_iou.info(verbose=True)
    
    # Offset for all possible anchors
    offset_list = np.empty(shape= anchor_list.shape, dtype= np.float32)
    offset_list.fill(0)
    offset_list[df_iou.anchor_id.values] = df_iou[["dx", "dy", "dw", "dh"]].values
    offset_list = np.expand_dims(offset_list, 0)
    
    # combine deltas and objectiveness score in one array
    offset_list_label_list = np.column_stack((offset_list[0], label_list[0]))[np.newaxis,:]
    
    #%% RPN def
    input_shape = (w_feature_map, h_feature_map, 512) # 50x50X512
    k = 1 # number of bbox by anchor ?
    input_ = Input(shape= input_shape)
    conv1 = Conv2D(512,
                   kernel_size= 3,
                   padding= "same",
                   name="conv1",
                   kernel_initializer=initializers.RandomNormal(stddev=0.01),
                   bias_initializer=initializers.Zeros())(input_) # (kw * iw + 2*padding_w / s_w) + 1
    
    # delta regression
    regressor = Conv2D(4*k,
                       kernel_size= 1,
                       activation= "linear",
                       name= "delta_regression",
                      kernel_initializer=initializers.RandomNormal(stddev=0.01),
                      bias_initializer=initializers.Zeros())(conv1) # (-1, 36)
    
    # objectiveness score
    classifier = Conv2D(k*1,
                        kernel_size= 1,
                        activation= "sigmoid",
                        name="objectivess_score",
                        kernel_initializer=initializers.RandomNormal(stddev=0.01),
                        bias_initializer=initializers.Zeros())(conv1)
    
    RPN = Model(inputs= [input_], outputs= [regressor, classifier])
    
    RPN.compile(loss = [custom_l1_loss, custom_binary_loss], optimizer= "adam")
    
    
    #%% RPN training
    
    history = RPN.fit(feature_maps,[offset_list_label_list, label_list], epochs= 40)
    plt.subplot(131)
    plt.title("loss")
    plt.plot(history.history["loss"])
    plt.subplot(132)
    plt.title("delta reg loss")
    plt.plot(history.history["delta_regression_loss"])
    plt.subplot(133)
    plt.title("objectivess loss")
    plt.plot(history.history["objectivess_score_loss"])
    plt.show()
    
    #%% RPN predict
    
    # get the offset and objectiveness score
    anchor_deltas, objectiveness_score = RPN.predict(feature_maps)
    
    first_conv=RPN.get_layer("conv1")
    obj_conv=RPN.get_layer("objectivess_score")
    x1=first_conv(feature_maps)
    x2=obj_conv(x1)
    plt.imshow(x2[0,:,:,0])
    plt.show()
    
    
    # shape both predictions
    anchor_deltas = anchor_deltas.reshape(-1, n_anchors, 4) # 50*50*9, 4
    objectiveness_score = objectiveness_score.reshape(-1, n_anchors) # 50*50*9, 1
    
    # parse anchor deltas
    dx = anchor_deltas[:, :, 0]
    dy = anchor_deltas[:, :, 1]
    dw = anchor_deltas[:, :, 2]
    dh = anchor_deltas[:, :, 3]
    
    print("shape of predicted anchor delta and objectiveness",anchor_deltas.shape, objectiveness_score.shape)
    
    #%% Filtering
    
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
    img_ = np.copy(img_for_plt)
    for i, bbox in enumerate(bbox_list):
        cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)
    
    for i in range(0, 10):  # 5625// 2
        #i=0
        #while score_sorted[0,i]>0.999:
        x_min = int(roi_sorted[i][0])
        y_min = int(roi_sorted[i][1])
        x_max = int(roi_sorted[i][2])
        y_max = int(roi_sorted[i][3])
        cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 
        print("confidence for",i,"th bbox :",score_sorted[0,i])
        i+=1
    
    
    plt.imshow(img_.astype(int))
    plt.show()
    
    #RPN.save("img202_ij_indexing.h5")
        
        
        
