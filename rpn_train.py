# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:53:19 2022

@author: Jean-Malo
"""
import numpy as np
import cv2
import glob
import json
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import pandas as pd
import time

# modifiaction of full_rpn script so that training is done on a data generator (using multiple images)
# only do training and save last model (can also use validation data)

# data generator struct based on https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
class DataGenerator(Sequence):
    '''Generates data for Keras'''
    def __init__(self, list_of_img, list_of_bbox_list, list_of_feature_maps=None, batch_size=1, shuffle=True):
        '''Initialization'''
        t=time.perf_counter()
        assert(len(list_of_bbox_list)==len(list_of_img))
        self.list_of_img = list_of_img
        self.list_of_bbox_list = list_of_bbox_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        #init backbone network
        vgg = keras.applications.VGG16(
            include_top=False,
            weights="imagenet"
        ) 
        self.backbone = Model(inputs= [vgg.layers[0].input], outputs= [vgg.layers[17].output])
        
        print("Backbone loaded",time.perf_counter()-t,"s")
        t=time.perf_counter()
        #init self.list_feature_maps : feature map of each input image
        self.list_of_feature_maps = list_of_feature_maps
        if self.list_of_feature_maps is None:
            self._init_feature_maps()
        #Our feature map dimension size is 50*50 (512 depth) = 2500, so total anchor centers per image will be 2500
        
        print("Feature maps computed",time.perf_counter()-t,"s")
        t=time.perf_counter()
        #init self.inside_anchor_list
        self._init_anchor_list()
        print("anchor computed",time.perf_counter()-t,"s")
        t=time.perf_counter()
        self._init_labels_and_offsets()
        print("labels and offsets computed",time.perf_counter()-t,"s")

        #self._init_offset()
        
        # init the order in wich images will be processed : self.indexes
        self.on_epoch_end() 
         
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_of_bbox_list) / self.batch_size))
    

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_of_bbox_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        # self.indexes give the order in witch images will be given at each epoch

    
    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        
        # print("indexes ",indexes)
        
        """
        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X
        """
        
        feature_maps = [self.list_of_feature_maps[k] for k in indexes]
        offsets = [self.list_of_offset_list_label_list[k] for k in indexes]
        labels = [self.list_of_label_list[k] for k in indexes]

        return feature_maps, [offsets, labels]
    
    def _init_feature_maps(self): 
        self.list_of_feature_maps=[]
        for img in self.list_of_img:
            t=time.perf_counter()
            self.list_of_feature_maps.append(self.backbone.predict(np.expand_dims(img, 0)))
            print(time.perf_counter()-t,"s")

    
    def _init_anchor_list(self):
        _, self.w_feature_map, self.h_feature_map, _ = self.list_of_feature_maps[0].shape
        img_h, img_w, _ = self.list_of_img[0].shape
        
        n_anchor_pos = self.w_feature_map * self.h_feature_map
        
        # width stride on the image
        x_stride = int(img_w / self.w_feature_map)
        y_stride = int(img_h / self.h_feature_map)
        
        # center (xy coordinate) of anchor location on image
        x_center = np.arange(8, img_w, x_stride) 
        y_center = np.arange(8, img_h, y_stride) 
        
        # generate all the ordered pair of x and y center
        
        # to achive this, we will use meshgrid and reshape it
        center_list = np.array(np.meshgrid(x_center, y_center,  sparse=False, indexing='ij')).T.reshape(-1,2)
        
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
        self.anchor_list = np.zeros(shape= (n_anchors, 4))
        
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
            self.anchor_list[count] = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]
            count += 1
        #print("anchor list shape",self.anchor_list.shape)
        
        h = img_h 
        w = img_w
        # select anchor boxes which are inside the image
        self.inside_anchor_idx_list = np.where(
            (self.anchor_list[:,0] >= 0) &
            (self.anchor_list[:,1] >= 0) &
            (self.anchor_list[:,2] <= w) &
            (self.anchor_list[:,3] <= h))[0]
        #print(self.inside_anchor_idx_list.shape)
        self.inside_anchor_list = self.anchor_list[self.inside_anchor_idx_list]
        #n_inside_anchor = len(self.inside_anchor_idx_list)
        
        #print("number of anchor \"inside\" the image", n_inside_anchor)
        
    def _init_labels_and_offsets(self):
        
        self.list_of_label_list = []
        self.list_of_offset_list_label_list = []
        n_inside_anchor=len(self.inside_anchor_idx_list)
        idx_img=0
        
        for bbox_list in self.list_of_bbox_list:
            n_object = len(bbox_list)
            
            # Computing IOU for each anchor boxes with each ground truth box 
            
            iou_list = np.zeros((n_inside_anchor, n_object))
            #print(iou_list.shape)
            # for each ground truth box
            for gt_idx, gt_box in enumerate(bbox_list):
                # for each anchor boxes
                for anchor_idx, anchor_box in enumerate(self.inside_anchor_list):
                    # compute IOU
                    iou_list[anchor_idx][gt_idx] = self._IOU(gt_box, anchor_box)
            # convert to dataframe
            
            # add anchor_id
            data = {"anchor_id" :self.inside_anchor_idx_list}
            
            # add object column and corresponding IOU
            data.update({f"object_{idx}_iou":iou_list[:, idx] for idx in range(n_object)})
            
            # for each anchor box assign max IOU among all objects in the image
            data["max_iou"] = iou_list.max(axis= 1)
            
            # for each anchorbox assign ground truth having maximum IOU
            data["best_gt"] = iou_list.argmax(axis= 1)
            
            df_iou = pd.DataFrame(data)
            
            #df_iou.info(verbose=True)
            
            
            # Remarkable values 
            # getting anchor boxes having maximum IOU for each ground truth boxes
            best_ious = df_iou.drop(["anchor_id", "max_iou", "best_gt"],axis= 1).max().values
            #print(f"Top IOUs for each object in the image: {best_ious}")
            
            # getting anchor box idx having maximum overlap with ground truth boxes * ignoring anchor id column
            best_anchors = df_iou.drop(["anchor_id", "max_iou", "best_gt"],axis= 1).values.argmax(axis= 0)
            #print(f"Top anchor boxes index: {best_anchors}")
            
            # get all the anchor boxes having same IOU score
            top_anchors = np.where(iou_list == best_ious)[0]
            #print(f"Anchor boxes with same IOU score: {top_anchors}")
    
            
            # visualizing top anchor boxes
            """
            img_ = np.copy(self.list_of_img[idx_img])
            for i in top_anchors:  # 5625// 2
                x_min = int(self.inside_anchor_list[i][0])
                y_min = int(self.inside_anchor_list[i][1])
                x_max = int(self.inside_anchor_list[i][2])
                y_max = int(self.inside_anchor_list[i][3])
                cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 
            
            for i, bbox in enumerate(bbox_list):
                cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=3)
            
            plt.imshow(img_)
            plt.show()
            idx_img+=1
            #"""
            
            # Labeling each anchor box based on IOU threshold 
            
            # create dummy column
            label_column = np.zeros(df_iou.shape[0], dtype= np.int32)
            label_column.fill(-1)
            
            # label top anchor boxes as 1 # contains object
            label_column[top_anchors] = 1
            
            # label anchor boxes having IOU > 0.4 with ground truth boxes as 1 # contains object
            label_column[np.where(df_iou.max_iou.values >= 0.4)[0]] = 1
            
            # label anchor boxes having IOU < 0.1 with ground truth boxes as 0 # background
            label_column[np.where(df_iou.max_iou.values < 0.1)[0]] = 0
            
            # add column to the iou dataframe
            df_iou["label"] = label_column
            
            #print("qty of foreground anchor",len(df_iou[df_iou.label == 1].index.values))
            #print("qty of background anchor",len(df_iou[df_iou.label == 0].index.values))
            self._sample_anchors_pre(df_iou)
            #print("qty of foreground anchor after sample",len(df_iou[df_iou.label == 1].index.values))
            #print("qty of background anchor after sample",len(df_iou[df_iou.label == 0].index.values))
            
            # Computing offset
            
            # for each valid anchor box coordinate, convert coordinate format
            inside_anchor_width, inside_anchor_height, inside_anchor_center_x, inside_anchor_center_y =  self._to_center_format(
                self.inside_anchor_list[:, 0],
                self.inside_anchor_list[:, 1],
                self.inside_anchor_list[:, 2],
                self.inside_anchor_list[:, 3])
            
            
            # for each ground truth box corresponds to each anchor box coordinate, convert coordinate format
            gt_coordinates = []
            for idx in df_iou.best_gt:
                gt_coordinates.append(bbox_list[idx])
            gt_coordinates = np.array(gt_coordinates)
            
            base_width, base_height, base_center_x, base_center_y =  self._to_center_format(
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
            
            n_anchors = self.w_feature_map * self.h_feature_map # true because we use only one scale and one ratio
            # labels for all possible anchors
            label_list = np.empty(n_anchors, dtype = np.float32)
            label_list.fill(-1)
            label_list[df_iou.anchor_id.values] = df_iou.label.values
            label_list = np.expand_dims(label_list, 0)
            label_list = np.expand_dims(label_list, -1)
            
            self.list_of_label_list.append(label_list)
            
            #df_iou.info(verbose=True)
            
            # Offset for all possible anchors
            offset_list = np.empty(shape= self.anchor_list.shape, dtype= np.float32)
            offset_list.fill(0)
            offset_list[df_iou.anchor_id.values] = df_iou[["dx", "dy", "dw", "dh"]].values
            offset_list = np.expand_dims(offset_list, 0)
            
            # combine deltas and objectiveness score in one array
            offset_list_label_list = np.column_stack((offset_list[0], label_list[0]))[np.newaxis,:]
        
            self.list_of_offset_list_label_list.append(offset_list_label_list)
        
        
        
    def _IOU(self, box1, box2):
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
    
    def _sample_anchors_pre(self, df, n_samples=1024, neg_ratio= 0.5):
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
            
    def _to_center_format(self, xmin_list, ymin_list, xmax_list, ymax_list):
        """
        Convert min max coordinate format to x_center, y_center, height and width format
        """
        height = ymax_list - ymin_list
        width = xmax_list - xmin_list
        
        center_x = xmin_list + 0.5 * width
        center_y = ymin_list + 0.5 * height
        
        return width, height, center_x, center_y

    def _adjust_deltas(self, anchor_width, anchor_height, anchor_center_x, anchor_center_y, dx, dy, dw, dh):
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

    def _compute_deltas(self, base_center_x, base_center_y, base_width, base_height, inside_anchor_width, inside_anchor_height, inside_anchor_center_x, inside_anchor_center_y):
        """
        computing offset of achor box to the groud truth box
        """
        dx = (base_center_x - inside_anchor_center_x)/ inside_anchor_width  # difference in centers of ground truth and anchor box across x axis
        dy = (base_center_y - inside_anchor_center_y)/  inside_anchor_height  # difference in centers of ground truth and anchor box across y axis
        dw = np.log(base_width/ inside_anchor_width) # log on ratio between ground truth width and anchor box width
        dh = np.log(base_height/ inside_anchor_height) # log on ratio between ground truth height and anchor box height
        return dx, dy, dw, dh


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
    y_pred = tf.reshape(y_pred, shape= (-1, 2500, 4)) #2500 is n_anchors (625 for 400x400 images)
    
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
    
    y_pred = tf.reshape(y_pred_objectiveness, shape= (-1, 2500)) #2500/625 is n_anchors
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

def init_gen(dir_path, load_fm=False):
    list_img_path = glob.glob(dir_path+"*.tif")
    list_json_path = glob.glob(dir_path+"bbox\\*.json")
    list_fm_path = glob.glob(dir_path+"featuremaps\\*.json")
    
    t=time.perf_counter()
    list_img=[]
    for path in list_img_path:
        list_img.append(cv2.imread(path))
    print("len list img",len(list_img),time.perf_counter()-t,"s")
    
    t=time.perf_counter()
    list_bboxes=[]
    for path in list_json_path:
        with open(path,'r') as file:
          list_bboxes.append(json.load(file))
    
    print("len bboxes list",len(list_bboxes),time.perf_counter()-t,"s")
    
    t=time.perf_counter()
    i=0
    for list_bbox in list_bboxes:
        i+=1
        #print("nb of bbox in image",i,":",len(list_bbox))
    
    list_fm=None
    
    if load_fm==True:
        list_fm=[]
        for path in list_fm_path:
            with open(path,'r') as file:
                fm=np.array(json.load(file))
                list_fm.append(fm)
        print("len list fm",len(list_fm),time.perf_counter()-t,"s")
        print("shape fm",list_fm[0].shape)

    gen=DataGenerator(list_img, list_bboxes, list_fm, batch_size=1)
    return gen
    
if __name__ == '__main__':
    # path to training images (with bbox folder)
    dir_path="./data/train/"
    
    gen=init_gen(dir_path,load_fm=True)
    
    # path to validation images (with bbox folder)
    dir_path="C:\\Users\\Jean-Malo\\Documents\\Polytech\\5A\\PRD_LiDAR\\test_scripts\\data_test\\LRMdataset\\rpn_data_val\\"
    
    #val_gen=init_gen(dir_path,load_fm=False)
    
    #"""
    #RPN def
    w_feature_map = h_feature_map = 50
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
    
    # if you want to resume training from an existing model
    #RPN = keras.models.load_model("rpn_1024_150e.h5", 
    #                              custom_objects={'custom_l1_loss':custom_l1_loss,'custom_binary_loss':custom_binary_loss})
    
    #create callback
    filepath = 'model\\rpn_data_val.h5' # best model on validation data
    checkpoint = ModelCheckpoint(filepath=filepath, 
                                 monitor='val_loss',
                                 verbose=1, 
                                 save_best_only=True,
                                 mode='min')
    callbacks = [checkpoint]
    
    # if you don't want to use validation data, comment the two last param
    history = RPN.fit_generator(gen, 
                                epochs=40)#,
    #                            callbacks=callbacks, validation_data=val_gen)
    
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
    
    """
    # plot loss curve on validation data
    plt.subplot(131)
    plt.title("val_loss")
    plt.plot(history.history["val_loss"])
    plt.subplot(132)
    plt.title("val_delta reg loss")
    plt.plot(history.history["val_delta_regression_loss"])
    plt.subplot(133)
    plt.title("val_objectivess loss")
    plt.plot(history.history["val_objectivess_score_loss"])
    plt.show()
    #"""
    RPN.save("model\\rpn_model.h5") # last model
    
    
    #"""