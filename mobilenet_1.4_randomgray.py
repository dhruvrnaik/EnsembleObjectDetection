#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:07:30 2019

@author: guest
"""

############################## check if first entry is being used corrctly or not !!!!!!!!!!!!!!!!!!!!! ###############
############################## GrayScale #######################################

from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import time
import torchvision.transforms as transforms
import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math
import numpy as np


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomGrayscale(),transforms.ToTensor()])

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
       
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        y = np.asarray([float(landmarks[0]),float(landmarks[1]),float(landmarks[2]),float(landmarks[3])])
        if self.transform:
            image = self.transform(image)
#            print(idx,"   ",y)
            y = torch.from_numpy(y).type(torch.FloatTensor)
        return [image,y]





bs = 16
vbs = 4
weight_name = "mobnetv2_1.4_rand_gray_smoothl1_{}.pth"
nwork = 4
epochs = 200
wt = "mobnetv2_1.4_rand_gray_smoothl1_37.pth"
start = 37

traindataset = FaceLandmarksDataset(csv_file='new_train.csv',
                                    root_dir='/media/guest/84B0DF27B0DF1F0C/aaa/oldie/FlipkartGridStage2DataSetImages/images/')

trainloader = DataLoader(traindataset, batch_size=bs,
                                          shuffle=True, num_workers=nwork)

testdataset = FaceLandmarksDataset(csv_file='new_val.csv',
                                    root_dir='/media/guest/84B0DF27B0DF1F0C/aaa/oldie/FlipkartGridStage2DataSetImages/images/')

testloader = DataLoader(testdataset, batch_size=vbs,
                                          shuffle=False, num_workers=nwork)

model_ft = MobileNetV2(4,width_mult=1.4).cuda()

net = model_ft

import torch.optim as optim

criterion = nn.SmoothL1Loss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=3)

min_valloss = np.inf

if wt!=None:
    state = torch.load(wt,map_location=lambda storage, loc: storage.cuda(0))
#    optimizer.load_state_dict(state['optimizer_state_dict'])
    net.load_state_dict(state['model_state_dict'])
    min_valloss=state['loss']

train_loss=[]
val_loss=[]

for epoch in range(1,epochs):  # loop over the dataset multiple times
    if epoch==1 or epoch==2:
        t = time.time()
    running_loss = 0.0
    running_loss_val=0.0
    
    net.train()

    for i, data in enumerate(trainloader, 1):
        
        optimizer.zero_grad()
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        
        outputs = net(inputs) 
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
#        outputs,iou = net(inputs,labels) 
#        iou_loss = torch.tensor(-1.0).cuda()*torch.log(iou+0.00000001)
#        loss = criterion(outputs, labels) + 1000*iou_loss
        
#        print(outputs)
        
#        iou_t += iou.item()
    avg_train_loss = running_loss/len(trainloader)
    print("Epoch- ",epoch+start," Loss- ",avg_train_loss)#," IoU- ",iou_t/len(trainloader))
    train_loss.append(avg_train_loss)
    if epoch==1 or epoch==2:
        print("Time -",time.time()-t)

    if (epoch+start)%51 == 0:
            print(" Weight saved ",epoch+start)
            torch.save({
            'epoch': epoch+start,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
            },weight_name.format(epoch+start))
    
    
    net.eval()   

    if epoch==1 or epoch==2:
        t = time.time()
        
    for i, data in enumerate(testloader, 1):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        outputs = net(inputs) 
       
        loss = criterion(outputs, labels)
        running_loss_val += loss.item()
#        print(i,"  ",outputs.item(),"  ",labels.item(),"  loss- ",loss,"   total_loss - ",running_loss_val)
#        if running_loss_val > 9999 or math.isnan(running_loss_val):
#            sys.exit()
#        
    
#        outputs,iou = net(inputs,labels) 
#            iou_loss = torch.tensor(-1.0).cuda()*torch.log(iou+0.00000000001)
#        loss = criterion(outputs, labels) #+ 1000*iou_loss
#        iou_v += func(outputs.detach().cpu(),labels.detach().cpu())
#        iou_loss = torch.tensor(-1.0).cuda()*torch.log(iou+0.00000001)
#        loss = criterion(outputs, labels) + 1000*iou_loss
#        iou_v += iou.item()
#            outputs = net(inputs)
#            loss = criterion(outputs, labels)
#            iou_v += iou.item()
    avg_val_loss = running_loss_val/len(testloader)
    
    scheduler.step(avg_val_loss)
    
    print("  Epoch- ",epoch+start,"Val Loss- ",avg_val_loss)#," IoU- ",iou_v/len(testloader))
    val_loss.append(avg_val_loss)
    if epoch==1 or epoch==2:
        print("Time -",time.time()-t)
        
    if avg_val_loss < min_valloss:
        min_valloss = avg_val_loss
        print("     Better weight found at ",epoch+start)
        torch.save({
        'epoch': epoch+start,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':avg_val_loss
        },weight_name.format(epoch+start))
#    if iou_v/len(testloader) > b_iou:
#        print("     Better iou found at ",epoch+start)
#        b_iou = iou_v/len(testloader)
#        torch.save({
#        'epoch': epoch+start,
#        'model_state_dict': net.state_dict(),
#        'optimizer_state_dict': optimizer.state_dict(),
#        'loss': running_loss_val,
#        'iou': b_iou,
#        },weight_b_iou)


print('Finished Training')
