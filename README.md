# rpn_helpers
Scripts on how to use Region Proposal Network to find charcoal kiln in LiDAR images

The architecture comes from https://github.com/martian1231/regionProposalNetworkInFasterRCNN/blob/master/region_proposal_network_in_keras_from_scratch.ipynb.

The notebook details nicely each step of a Region Proposal Network.

###### seg_to_bbox.py
Region Proposal Network use bounding boxes, but that data i had was for segmentation : each LiDAR image was associated with a mask. So i used this script to transform mask into list of bounding boxes. The mask image should have a white background and the object should be black. For each image a list of bbox is generated using connected component algorithm, this list is saved in a json file with the same base name as the image. All the json files are saved in a directory named bbox (created in the directory containing the images).
The directory containing the images should be given directly in the code. (for each script i used absolute link)
Once the bbox_dir is created, it shoulde be copied in the corresponding image directory.


- img_dir
- - bbox
- - - img1.json
- - - img2.json
- - - ...
- - img1.png
- - img2.png
- - ...

###### verif_bbox.py
A json file can't be easly visualized, so this script take in input a mask directory and an image directory and plot the original image with drawn bbox next to the corresponding mask (for each image). Drawn bbox are coming from the bbox directory in the image directory (not the mask directory). Again, each path is absolute and have to be modified directly in the code.

###### full_rpn.py
It's an adaptation of the notebook in a .py file, that use one LiDAR image for training and predict the result on this same image. It's usefull to see the effects of different parameters :
- n_sample : defines the number of sample / patch used to do the training. In the notebook it's 256, but because we search small objects, i noticed better results when increasing this value to 1024.
- anchor_ratio_list and anchor_scale_list : defined the default size of patches. The charcoal kilns are quite consitante in size so i don't use a list of value : each patch is a square (good results with 35x35)
- epochs : 40 is enough for one image, a loss graph is ploted
- visualizing top anchor boxes : here we can either plot the x top boxes or plot all the boxes with a confidance score higher than x

###### rpn_train.py
The goal of this script is to use multiple images to train the RPN (using keras data_generator). The input should be an image directory containing a bbox directory (as defined in the seg_to_bbox.py part). RPN transform each image into feature maps using a backbone network (here vgg16, pretrained on imagenet), this operations is quite long so you can give a precomputed feature map folder (same principle as the bbox directory). See img_to_featuremap.py for more info. This script can use validation data (even if not really efficient on my charcoal kiln data). You have to define manually the name of the saved model, it will be saved in the current directory. You can also reload a model befor starting the training. Be careful this script can use a lot of memormy if you give a folder with a lot of images (i have a computer with 8GB of RAM, and when running with 800 LiDAR images i lose 10GB of memory on my SSD, rebbooting my computer gives me back my 10GB)

###### img_to_featuremap.py
Take in input an image directory, each image is passed to vgg16 convolution layers, the result is saved as json file (same base name as the image) in a featuremap directory (created in the image directory). This allow a quicker training time (3s to compute vs 0.5 to read).

###### 800to4x400.py
Scipt that transform 1 images in 4 image of reduce size : so the charcoal kiln are bigger. Not really useful because the total size of the image gives the number of default anchor used in RPN and if this number is smaller we lose precision.
