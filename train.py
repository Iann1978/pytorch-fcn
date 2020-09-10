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


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

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
        return x_train[0,:,0:640,0:640].to(device),y_train[0,:,0:640,0:640].to(device)
    

     
     

def fit(eporchs, model, criterion, optimizer,train_dl, valid_dl,debug=False):
    for t in range(eporchs):
        for x_train,y_train in train_dl:
            y_pred = model(x_train)
            
            print(y_pred.shape)
            
            

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
        #self.fc6 = Conv(512,4096,3,1)
        #self.drop6 = nn.Dropout2d()
        
        # fc7
        #self.fc7 = Conv(4096,4096,3,1)
        #self.drop7 = nn.Dropout2d()
        
        self.score_fr =  Conv(512,1,3,1)
        self.score_pool3 = Conv(256, 1, 3, 1)
        self.score_pool4 = Conv(512, 1, 3, 1)
        
        #self.upscore1 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upscore1 = nn.Upsample(scale_factor=2)
        self.upscore2 = nn.Upsample(scale_factor=2)
        self.upscore3 = nn.Upsample(scale_factor=2)
        
        
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
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        
        #x = self.fc6(x)
        #x = self.drop6(x)
        
        #x = self.fc7(x)
        #x = self.drop7(x)
        
        x = self.score_fr(x)
        
        fc7_score = x
        y = self.upscore1(fc7_score)
        
        x = self.upsam1(x)
        x = x + y
        #x = y
        x = self.upsam2(x)
        x = self.upsam3(x)
        x = self.upsam4(x)
        x = self.upsam5(x)
        return x


def train(opt):
    
    epochs = opt.epochs
    weights = opt.weights
    
    train_ds = SentimentDataset()
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1,shuffle=False)
    
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
    
    return

    
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
 
