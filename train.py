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

from torch.utils.tensorboard import SummaryWriter

from models.common import Conv
from models.fcn import Net

from utils.datasets import SentimentDataset
from utils.datasets import BagDataset

device = torch.device("cuda:0")

# x_train = torch.Tensor(np.random.randn(1,3,50,50))
# y_train = torch.Tensor(np.random.randn(1,3,50,50))

# x_train0 = cv2.imread("./data/22678915_15.tif")
# x_train1 = x_train0.transpose(2,0,1)
# x_train2 = x_train1[np.newaxis,:]
# x_train3 = x_train2.astype(np.float32)
# x_train4 = x_train3/255.0;
# x_train = torch.Tensor(x_train4)


# y_train0 = cv2.imread("./data/22678915_15.tiff")
# y_train1 = y_train0.transpose(2,0,1)
# y_train2 = y_train1[np.newaxis,:]
# y_train3 = y_train2.astype(np.float32)
# y_train4 = y_train3/255.0;
# y_train = torch.Tensor(y_train4)




#cv2.imshow("x_train0", x_train0)
#cv2.imshow("y_train1", y_train1)



def fit(eporchs, model, criterion, optimizer,train_dl, valid_dl,debug=False):
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
            print(t, loss.item())
            
                # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




def train(opt):
    
    epochs = opt.epochs
    weights = opt.weights
    batch_size = opt.batch_size
    img_size = tuple(opt.img_size)
    
    #train_ds = SentimentDataset(device)
    train_ds = BagDataset('./data/two_bag',device=device,img_size=img_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,shuffle=False)
    
    for xb,yb in train_dl:
         print(xb.shape)
         print(yb.shape)
         
    #if os.path.exists(test_file.txt)
    #model = Net()
    model = torch.load(weights) if os.path.exists(weights) else Net()
    model.to(device)
   
    # Output model to tensorboard
    images, masks = next(iter(train_dl))    
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    writer.add_graph(model, images)
    writer.close()
    
   

    
    criterion =  lambda y_pred, y_true: torch.square(y_true-y_pred).sum()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-9)
    fit(epochs, model,criterion, optimizer,train_dl,None,debug=False)

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
    
    return

    
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='train,test sizes')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    opt = parser.parse_args()
    print(opt)
    
    
    

    train(opt)

#train()
# def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
#     for epoch in range(epochs):
#         for xb,yb in train_dl:
#             pred = model(xb)
#             loss = loss_func(pred,yb)
#             #print (loss)
            
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
            
#         model.eval()
#         with torch.no_grad():
#             valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

#         print(epoch, valid_loss / len(valid_dl))

# o = cv2.imread("./data/22678915_15.tif")


# cv2.imshow("origin", o)
# #cv2.imshow("erode result", r)
# cv2.waitKey()
# cv2.destroyAllWindows()

# oo = o.transpose(2,0,1)
# ooo = oo[np.newaxis,:]


# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, D_out = 4, (5,5), (5,5)

# inputShape = [N] + list(D_in)
# outputShape = [N] + list(D_out)

# # Create random input and output data
# x = np.random.randn(*inputShape)
# y = np.random.randn(*outputShape)

# # Randomly initialize weights
# w1 = np.random.randn(25, 25)

# learning_rate = 1e-6

# for t in range(2000):
#     xx = x.reshape(4,25)
#     yy = y.reshape(4,25)
#     yy_pred = xx.dot(w1)
    
#     y_pred = yy_pred.reshape(4,5,5)
    
#     loss = np.square(y_pred-y).sum()   
#     print(t, loss) 
    
#     grad_yy_pred = 2.0 * (yy_pred - yy)
#     grad_w1 = xx.T.dot(grad_yy_pred)
#     w1 -= learning_rate * grad_w1
 
