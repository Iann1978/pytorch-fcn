# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:51:25 2020

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
import yaml
import shutil

from torch.utils.tensorboard import SummaryWriter

from models.common import Conv
from models.fcn import Net


from utils.datasets import SentimentDataset
from utils.datasets import BagDataset

device = torch.device("cuda:0")





def fit(eporchs, model, criterion, optimizer,train_dl, valid_dl, writer=None,debug=False):
    print('')
    for t in range(eporchs):
        
       
        for x_train,y_train in train_dl:
            y_pred = model(x_train)
            
            #print(y_pred.shape)
            
            

            if (debug):
                pic_pred = y_pred.detach().numpy()
                pic_pred = pic_pred.squeeze()
                #pic_pred = pic_pred.transpose(1,2,0)
                print(pic_pred.shape)
                cv2.imshow("pic_pred", pic_pred)
            #cv2.imshow("erode result", r)
            #cv2.waitKey()
            #cv2.destroyAllWindows()

            
            # Compute and print loss
            loss = criterion(y_pred, y_train)
                
            #if t % 10 == 9:
            print('\r',  t, loss.item(),end='')
            


            
            
                # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        # Evaluation
        losses = []
        for x_valid, y_valid in valid_dl:
             y_pred_valid = model(x_valid)
             loss = criterion(y_pred_valid, y_valid)
             losses.append(loss.item())
        average_loss = np.mean(losses)
        #print('--------------------------------')
        print(t, average_loss.item())
        #print('--------------------------------')
        
        writer.add_scalar('valid loss', average_loss, t)



def train(opt):
    
    epochs = opt.epochs
    weights = opt.weights
    batch_size = opt.batch_size
    img_size = tuple(opt.img_size)
    data_path = opt.data_path
    
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    
    #train_ds = SentimentDataset(device)
    train_ds = BagDataset(data_dict['train'],device=device,img_size=img_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,shuffle=False)
    valid_ds = BagDataset(data_dict['val'],device=device,img_size=img_size)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size,shuffle=False)
    
    for xb,yb in train_dl:
         print(xb.shape)
         print(yb.shape)
         break
         
    #if os.path.exists(test_file.txt)
    #model = Net()
    model = torch.load(weights) if os.path.exists(weights) else Net()
    #model = Model(cfg='models/fcn8s.yaml')
    model.to(device)
   
    # Output model to tensorboard
    images, masks = next(iter(train_dl))  
    if os.path.exists('runs/fashion_mnist_experiment_1'):
        shutil.rmtree('runs/fashion_mnist_experiment_1')
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    writer.add_graph(model, images)
    #writer.close()
    
   

    
    criterion =  lambda y_pred, y_true: torch.square(y_true-y_pred).sum()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-9)
    fit(epochs, model, criterion, optimizer, train_dl, valid_dl, 
        writer=writer, debug=False)

    # Save model    
    torch.save(model, './inference/model')
    
    
     # Save the predict
    print('images.shape:', images.shape)
    pred = model(images);
    pred1 = pred.cpu().detach().numpy();
    pred2 = pred1[0,:]
    pred3 = pred2.transpose(1,2,0)
    pred4 = pred3*255.
    print('pred4.shape:', pred4.shape)
    cv2.imwrite('./inference/pred.jpg', pred4)
    
    topred1= images.cpu().detach().numpy()
    topred2 = topred1[0,:]
    topred3 = topred2.transpose(1,2,0)
    topred4 = topred3*255.
    cv2.imwrite('./inference/topred4.jpg', topred4)
    writer.close()
    return

    
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/two_bags', help='data path')
    parser.add_argument('--data', type=str, default='data/two_bags.yaml', help='data.yaml path')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='train,test sizes')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    opt = parser.parse_args()
    print(opt)
    

    
    train(opt)
