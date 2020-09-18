# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:36:04 2020

@author: Iann
"""


import torch.nn as nn
import torch
import torch.utils.data
from models.common import Conv
import yaml
from pathlib import Path
from copy import deepcopy
import argparse

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # conv1
        self.conv1_1 = Conv(3,64,3,1)
        self.conv1_2 = Conv(64,64,3,1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 1/2
        
        
        # conv2
        self.conv2_1 = Conv(64,128,3,1)
        self.conv2_2 = Conv(128,128,3,1)
        self.pool2 = nn.MaxPool2d(2,2)  # 1/4
        
        # conv3
        self.conv3_1 = Conv(128,256,3,1)
        self.conv3_2 = Conv(256,256,3,1)
        self.pool3 = nn.MaxPool2d(2,2)  # 1/8
        
        # conv4
        self.conv4_1 = Conv(256,512,3,1)
        self.conv4_2 = Conv(512,512,3,1)
        self.conv4_3 = Conv(512,512,3,1)
        self.pool4 = nn.MaxPool2d(2,2)  # 1/16
        
        # conv5
        self.conv5_1 = Conv(512,512,3,1)
        self.conv5_2 = Conv(512,512,3,1)
        self.conv5_3 = Conv(512,512,3,1)
        self.pool5 = nn.MaxPool2d(2,2)  # 1/32
        
        # fc6
        self.fc6 = Conv(512,512,19,1)
        self.drop6 = nn.Dropout2d()
        
        # fc7
        self.fc7 = Conv(512,512,19,1)
        #self.drop7 = nn.Dropout2d()
        
        self.score_32s =  Conv(512,1,3,1)
        self.score_16s = Conv(1024, 1, 3, 1)
        self.score_8s = Conv(1280, 1, 3, 1)
        
        self.upscore1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        #self.upscore1 = nn.Upsample(scale_factor=2)
        self.upscore2 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1, bias=False)
        self.upscore3 = nn.ConvTranspose2d(1280, 1280, 4, 2, 1, bias=False)
        
        
        self.upsam1 = nn.Upsample(scale_factor=2)
        self.upsam2 = nn.Upsample(scale_factor=2)
        self.upsam3 = nn.Upsample(scale_factor=2)
        self.upsam4 = nn.Upsample(scale_factor=2)
        self.upsam5 = nn.Upsample(scale_factor=2)

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        pool3 = x  # 1/8
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        pool4 = x  # 1/16
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        
        x = self.fc6(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        #x = self.drop7(x)
        fc7 = x
        
        x = self.upscore1(fc7)
        
        fcn_32s = self.score_32s(x) 
        

        x = torch.cat((pool4,x),1)
        x = self.upscore2(x)
        
        fcn_16s = self.score_16s(x)
        
        #print('pool3.shape:', pool3.shape)
        #print('x.shape:', x.shape)
        x = torch.cat((pool3,x),1)
        
        x = self.upscore3(x)
        fcn_8s = self.score_8s(x)
         
        x = fcn_8s
        #fc7_score = x
        
       
        # = self.upsam1(x)
        
        #y = self.upscore1(fc7_score)
        #x = x + y
        #x = y
        #x = self.upsam2(x)
        #x = self.upsam3(x)
        x = self.upsam4(x)
        x = self.upsam5(x)
        return x

