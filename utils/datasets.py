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
      
     