# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 09:20:54 2020

@author: Iann
"""

import argparse
import os
import torch
import cv2
import numpy as np

def detect(opt):
    source = opt.source
    weights = opt.weights
    
    if not os.path.exists(weights):
        return 
    device = torch.device("cuda:0")
    model = torch.load(weights) 
    model.to(device)
    
    x_train0 = cv2.imread(source)
    x_train0 = cv2.resize(x_train0,(640,640))
    x_train1 = x_train0.transpose(2,0,1)
    x_train2 = x_train1[np.newaxis,:]
    x_train3 = x_train2.astype(np.float32)
    x_train4 = x_train3/255.0;
    x_train = torch.Tensor(x_train4)
    x_train = x_train.to(device)
    
    pred = model(x_train)
    pred = pred.cpu().detach().numpy()
    pred = pred[0,:]
    pred = pred.transpose(1,2,0)
    pred = pred*255.
    cv2.imwrite('./runs/pred.jpg', pred)
    
    
    
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/bags/valuation/images/0.jpg', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--data-path', type=str, default='./data/two_bags', help='data path')
    # parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    # parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='train,test sizes')
    # parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='initial weights path')
    opt = parser.parse_args()
    print(opt)
    

    
    detect(opt)