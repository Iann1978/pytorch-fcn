# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:51:25 2020

@author: Iann
"""



import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torch.utils.data

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
    

     
     

def fit(eporchs, model, criterion, optimizer,dataloader,debug=False):
    for t in range(eporchs):
        for x_train,y_train in dataloader:
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
        
        self.conv1_1 = nn.Conv2d(3, 64, 3,padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3,padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 1/2
        
        
         # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 1, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2,2)  # 1/4
        
        
        self.upsam1 = nn.Upsample(scale_factor=2)
        self.upsam2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        
        x = self.upsam1(x)
        x = self.upsam2(x)
        return x


def train():
    dataset = SentimentDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False)
    
    for xb,yb in dataloader:
         print(xb.shape)
         print(yb.shape)
         
    
    model = Net()
    model.to(device)
    
    criterion =  lambda y_pred, y_true: torch.square(y_true-y_pred).sum()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-9)
    
    
    
    fit(20, model,criterion, optimizer,dataloader,debug=False)

train()
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
 
