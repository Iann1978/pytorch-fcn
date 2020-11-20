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
from utils.general import increment_dir, get_latest_run

from models.common import Conv
from models.fcn import Net


from utils.datasets import SentimentDataset
from utils.datasets import BagDataset
from utils.datasets import CocoDataset

device = torch.device("cuda:0")


def iou(y_train0, y_pred0, debug=False):
    y_train1 = 1 - y_train0
    y_pred1 = 1 - y_pred0
    ret,y_pred2 = cv2.threshold(y_pred1,0.5,1,cv2.THRESH_BINARY)
    y_pred3 = y_pred2[:,:,np.newaxis]
    
    cal_intercetion = np.vectorize(lambda x,y:x*y)
    cal_union = np.vectorize(lambda x,y:max(x,y))
    
    intercetion = cal_intercetion(y_pred3,y_train1)
    union = cal_union(y_pred3,y_train1)

    if debug:    
        cv2.imshow("y_train1", y_train1)
        cv2.imshow("y_pred2",y_pred2)
        cv2.imshow("intercetion",intercetion)
        cv2.imshow("union",union)
        cv2.waitKey()
    
    intercetion_score = np.sum(intercetion)
    union_score = np.sum(union)
    iou_score = intercetion_score/union_score
    if debug:    
        print("intercetion_score:",intercetion_score)
        print("union_score",union_score)
        print("iou_score", iou_score)
    return iou_score


def fit(eporchs, model, criterion, optimizer,train_dl, valid_dl, writer=None,debug=False):
    print('')
    
    # Get work dir
    if writer:
        wdir = writer.log_dir
    else:
        wdir = increment_dir('./runs/exp')
        os.mkdir(wdir)

    # Initialize average loss        
    last_average_loss = float("inf")


    # Start fit
    print('eporch,                loss,       average_loss,         mean-iou')
    for t in range(eporchs):
        
       
        for x_train,y_train in train_dl:
            y_pred = model(x_train)
            
            

            # Debug show y_train0
            y_train0 = y_train.cpu().detach().numpy()[0];
            #src0 *= 255.0
            y_train0 = y_train0.transpose(1,2,0)
            #cv2.imshow("y_train0", y_train0)
            #cv2.waitKey()
            #print(y_pred.shape)
            
            
            # Debug show y_pred
            y_pred0 = y_pred.cpu().detach().numpy()[0];
            y_pred0 = y_pred0.transpose(1,2,0)
            #cv2.imshow("y_pred0", y_pred0)
            #cv2.waitKey()
            
            
            #iou(y_train0, y_pred0)
            # if (debug):
            #     y_pred0 = y_pred.cpu().detach().numpy()[0];
            #     y_pred0 = y_pred0.transpose(1,2,0)
            #     #pic_pred = y_pred0.transpose(1,2,0)
            #     print(pic_pred.shape)
            #     cv2.imshow("pic_pred", pic_pred)
            #cv2.imshow("erode result", r)
            #cv2.waitKey()
            #cv2.destroyAllWindows()

            
            # Compute and print loss
            loss = criterion(y_pred, y_train)
                
            #if t % 10 == 9:
            print('\r%6d,%20f,'%(t, loss.item()),end='')
            


            
            
                # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        # Evaluation
        losses = []
        ious = []
        for x_valid, y_valid in valid_dl:
            y_pred_valid = model(x_valid)
            loss = criterion(y_pred_valid, y_valid)
            losses.append(loss.item())
             
            y_train0 = y_valid.cpu().detach().numpy()[0];
            y_train0 = y_train0.transpose(1,2,0)
            y_pred0 = y_pred_valid.cpu().detach().numpy()[0];
            y_pred0 = y_pred0.transpose(1,2,0)
            
            iou_score =  iou(y_train0, y_pred0)
            ious.append(iou_score)
            
           
            
        average_loss = np.mean(losses)
        mean_iou = np.mean(ious)
        #print('--------------------------------')
        print('%20f,%20f'% (average_loss.item(),mean_iou.item()))
        #print('--------------------------------')
        

        # Save best and last
        weights_dir = os.path.join(wdir,'weights')
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
        best = os.path.join(weights_dir,'best.pt')
        last = os.path.join(weights_dir,'last.pt')
        torch.save(model, last)        
        if average_loss < last_average_loss:
            torch.save(model, best)
            last_average_loss = average_loss
        
        # Write to tensorboard
        writer.add_scalar('valid loss', average_loss, t)

 
    
    

def train(opt):
    
    epochs = opt.epochs
    weights = opt.weights
    batch_size = opt.batch_size
    img_size = tuple(opt.img_size)
    data_path = opt.data_path
    
    # Create work dir
    wdir = increment_dir('./runs/exp')
    os.mkdir(wdir)
    
    # Load data
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    
    #train_ds = SentimentDataset(device)
    #train_ds = BagDataset(data_dict['train'],device=device,img_size=img_size)
    train_ds = CocoDataset(data_dict['train'],device=device,img_size=img_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,shuffle=False)
    #valid_ds = BagDataset(data_dict['val'],device=device,img_size=img_size)
    valid_ds = CocoDataset(data_dict['val'],device=device,img_size=img_size)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size,shuffle=False)
    
    for xb,yb in train_dl:
         print(xb.shape)
         print(yb.shape)
         break
         
     
    # Load model
    #if os.path.exists(test_file.txt)
    #model = Net()
    model = torch.load(weights) if os.path.exists(weights) else Net()
    #model = Model(cfg='models/fcn8s.yaml')
    model.to(device)
    
    
   
    # Output model to tensorboard
    images, masks = next(iter(train_dl))  
    writer = SummaryWriter(wdir)
    writer.add_graph(model, images)
    #writer.close()
    
   

    
    criterion =  lambda y_pred, y_true: torch.square(y_true-y_pred).sum()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-9)
    fit(epochs, model, criterion, optimizer, train_dl, valid_dl, 
        writer=writer, debug=True)

    # Save model    
    print('wight file has been saved to ./inference/model')
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
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/last.pt, or most recent run if blank')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    opt = parser.parse_args()
    print(opt)
    
    # latest = get_latest_run()

    # print(latest)
    
    
    # Resume
    last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # resume from most recent run
    if last and not opt.weights:
        print(f'Resuming training from {last}')
    opt.weights = last if opt.resume and not opt.weights else opt.weights
    print(opt)
     
    train(opt)
