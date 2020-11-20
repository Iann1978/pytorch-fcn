# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:46:33 2020

@author: Iann
"""

import argparse
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torch.utils.data
import os

from torch.utils.tensorboard import SummaryWriter




   
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self,device):
        super().__init__()
        self.device = device

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        x_train0 = cv2.imread("./data/22678915_15.tiff")
        x_train1 = x_train0.transpose(2,0,1)
        x_train2 = x_train1[np.newaxis,:]
        x_train3 = x_train2.astype(np.float32)
        x_train4 = x_train3/255.0;
        x_train = torch.Tensor(x_train4)
        
        
        y_train0 = cv2.imread("./data/22678915_15.tif")
        y_train1 = y_train0.transpose(2,0,1)
        y_train2 = y_train1[np.newaxis,:]
        y_train3 = y_train2.astype(np.float32)
        y_train4 = y_train3/255.0;
        y_train = torch.Tensor(y_train4)
        
        y_train0 = cv2.imread("./data/22678915_15.tif")
        y_train1 = y_train0[:,:,2]
        y_train2 = y_train1[np.newaxis, np.newaxis,:]
        y_train3 = y_train2.astype(np.float32)
        y_train4 = y_train3/255.0;
        y_train = torch.Tensor(y_train4)
        
        #cv2.imshow("x_train0", x_train0)
        #cv2.imshow("y_train0", y_train0)
        #cv2.imshow("y_train1", y_train1)
        return x_train[0,:,0:640,0:640].to(self.device),y_train[0,:,0:640,0:640].to(self.device)
      


class BagDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,device,img_size=(320,320)):
        super().__init__()
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir,"images")
        self.msk_dir = os.path.join(data_dir,"masks")
        self.img_size = img_size
        self.device = device
        #self.img_count = len([lists for lists in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, lists))])
        
        self.images = [f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir,f))]
        self.masks =  [f for f in os.listdir(self.msk_dir) if os.path.isfile(os.path.join(self.msk_dir,f))]
        self.images.sort()
        self.masks.sort()
        assert len(self.images) == len(self.masks)
        self.img_count = len(self.images)

    def __len__(self):
        return self.img_count
        
    def __getitem__(self, idx):
        
        #img_file = os.path.join(self.img_dir,"%d.jpg"%idx)
        #msk_file = os.path.join(self.msk_dir,"%d.jpg"%idx)
        
        img_file = os.path.join(self.img_dir,self.images[idx])
        msk_file = os.path.join(self.msk_dir,self.masks[idx])
        
        x_train0 = cv2.imread(img_file)
        x_train0 = cv2.resize(x_train0, self.img_size)
        x_train1 = x_train0.transpose(2,0,1)
        x_train2 = x_train1[np.newaxis,:]
        x_train3 = x_train2.astype(np.float32)
        x_train4 = x_train3/255.0;
        x_train = torch.Tensor(x_train4)
        #cv2.imshow("BagDataset.x_train0",x_train0)
        #cv2.waitKey()
        
        
        y_train0 = cv2.imread(msk_file)
        y_train0 = 255-y_train0
        y_train0 = cv2.resize(y_train0, self.img_size)
        y_train1 = y_train0[:,:,2]
        y_train2 = y_train1[np.newaxis, np.newaxis,:]
        y_train3 = y_train2.astype(np.float32)
        y_train4 = y_train3/255.0;
        y_train = torch.Tensor(y_train4)
        
                #cv2.imshow("BagDataset.x_train0",x_train0)
        #
        
        #cv2.imshow("x_train0", x_train0)
        #cv2.imshow("y_train0", y_train0)
        #cv2.imshow("y_train1", y_train1)
        return x_train[0,:,0:640,0:640].to(self.device),y_train[0,:,0:640,0:640].to(self.device)
      
from pycocotools.coco import COCO
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,device,img_size=(320,320)):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.device = device
       
        dataDir='G:/datasets/coco/2017'
        dataType='val2017'
        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
        
        # initialize COCO api for instance annotations
        coco=COCO(annFile)

        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = annFile
        self.coco = coco
        self.catIds = coco.getCatIds(catNms=['dog'])
        self.imgIds = coco.getImgIds(catIds=self.catIds )
        self.imgs = coco.loadImgs(self.imgIds)
        self.imgs =  self.imgs[0:15]


    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        img = self.imgs[idx]
        coco = self.coco
        catIds = self.catIds
        dataDir = self.dataDir
        dataType = self.dataType
        
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        anns_img = np.zeros((img['height'],img['width']))
        
        imgfile = '%s/%s/%s'%(dataDir,dataType,img['file_name'])
        src_img = cv2.imread(imgfile)
        cv2.imshow("src_img",src_img)
        #cv2.waitKey()
    
        ann = anns[0]
        
        print(ann)
        anns_img1 = coco.annToMask(ann)
        print(anns_img1.shape)
        anns_img11 = anns_img1 * 255
        anns_img2 = coco.annToMask(ann)*ann['category_id']
        anns_img3 = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])
        cv2.imshow("anns_img",anns_img)
        cv2.imshow("anns_img1", anns_img1)
        cv2.imshow("anns_img11", anns_img11)
        cv2.imshow("anns_img2", anns_img2)
        cv2.imshow("anns_img3", anns_img3)
        #cv2.waitKey()
        
        #img_file = os.path.join(self.img_dir,"%d.jpg"%idx)
        #msk_file = os.path.join(self.msk_dir,"%d.jpg"%idx)
        
        #img_file = os.path.join(self.img_dir,self.images[idx])
        #msk_file = os.path.join(self.msk_dir,self.masks[idx])
        
        x_train0 = src_img
        #x_train0 = cv2.imread(img_file)
        x_train0 = cv2.resize(x_train0, self.img_size)
        x_train1 = x_train0.transpose(2,0,1)
        x_train2 = x_train1[np.newaxis,:]
        x_train3 = x_train2.astype(np.float32)
        x_train4 = x_train3/255.0;
        x_train = torch.Tensor(x_train4)
        #cv2.imshow("BagDataset.x_train0",x_train0)
        #cv2.waitKey()
        
        #y_train0 = anns_img11
        #y_train0 = cv2.imread(msk_file)
        #y_train0 = 255-y_train0
        #y_train0 = cv2.resize(y_train0, self.img_size)
        #y_train1 = anns_img11
        y_train1 = cv2.resize(anns_img11, self.img_size)
        y_train2 = y_train1[np.newaxis, np.newaxis,:]
        y_train3 = y_train2.astype(np.float32)
        y_train4 = y_train3/255.0;
        y_train = torch.Tensor(y_train4)
        
                #cv2.imshow("BagDataset.x_train0",x_train0)
        #
        
        cv2.imshow("x_train0", x_train0)
        cv2.imshow("y_train1", y_train1)
        #cv2.waitKey()
        #cv2.imshow("y_train1", y_train1)
        return x_train[0,:,0:640,0:640].to(self.device),y_train[0,:,0:640,0:640].to(self.device)
      