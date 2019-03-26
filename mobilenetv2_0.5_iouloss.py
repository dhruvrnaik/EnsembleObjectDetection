#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:53:08 2019

@author: guest
"""

################ with iou  loss

from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn

import torchvision.transforms as transforms
import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math
import numpy as np

def func(yy,ll):
    ious = torch.tensor(0.0)
    #print(yy.size()[0])
    yy = torch.index_select(yy,1,torch.LongTensor([0,2,1,3]))
    ll = torch.index_select(ll,1,torch.LongTensor([0,2,1,3]))
    for i in range(yy.size()[0]):
        y = yy[i]
        l = ll[i]
        z = torch.stack((y,l))
        maxi,_ = torch.max(z,0)
        mini,_ = torch.min(z,0)
        interArea = torch.max(torch.tensor(0.0),mini[2]-maxi[0]+torch.tensor(1.0))*torch.max(torch.tensor(0.0),mini[3]-maxi[1]+torch.tensor(1.0))
#            print("interArea- ",interArea)
        boxAArea = (y[2]-y[0])*(y[3]-y[1])
        boxBArea = (l[2]-l[0])*(l[3]-l[1])
        
        iou = (interArea)/(boxAArea+boxBArea-interArea)
        ious+=iou
    #yy = torch.index_select(yy,1,torch.LongTensor([0,2,1,3]))
    return ious/float(yy.size()[0])

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
        
#    def intersect(self,box_a, box_b):
#        A = box_a.size(0)
#        B = box_b.size(0)
#        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
#                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
#        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
#                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
#        inter = torch.clamp((max_xy - min_xy), min=0)
#        return inter[:, :, 0] * inter[:, :, 1]
#    
#    
#    def jaccard(self,box_a, box_b):
#        box_a = torch.index_select(box_a,1,torch.cuda.LongTensor([0,2,1,3]))
#        box_b = torch.index_select(box_b,1,torch.cuda.LongTensor([0,2,1,3]))
#        inter = self.intersect(box_a, box_b)
#        area_a = ((box_a[:, 2]-box_a[:, 0]) *
#                  (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#        area_b = ((box_b[:, 2]-box_b[:, 0]) *
#                  (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#        union = area_a + area_b - inter
#        box_a = torch.index_select(box_a,1,torch.cuda.LongTensor([0,2,1,3]))
#        return inter / union # [A,B]   SEE MEE
        
    
    def func(self,yy,ll):
        ious = torch.tensor(0.0).cuda()
        #print(yy.size()[0])
        yy = torch.index_select(yy,1,torch.cuda.LongTensor([0,2,1,3]))
        ll = torch.index_select(ll,1,torch.cuda.LongTensor([0,2,1,3]))
        for i in range(yy.size()[0]):
            y = yy[i]
            l = ll[i]
            z = torch.stack((y,l))
            maxi,_ = torch.max(z,0)
            mini,_ = torch.min(z,0)
            interArea = torch.max(torch.tensor(0.0).cuda(),mini[2]-maxi[0]+torch.tensor(1.0).cuda())*torch.max(torch.tensor(0.0).cuda(),mini[3]-maxi[1]+torch.tensor(1.0).cuda())
#            print("interArea- ",interArea)
            boxAArea = (y[2]-y[0])*(y[3]-y[1])
            boxBArea = (l[2]-l[0])*(l[3]-l[1])
            
            iou = (interArea)/(boxAArea+boxBArea-interArea)
            ious+=iou
        yy = torch.index_select(yy,1,torch.cuda.LongTensor([0,2,1,3]))
        return ious/float(yy.size()[0])

    def forward(self, x, labels):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        iou = self.func(x,labels)
        return x,iou

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
        self.transform = transforms.Compose([transforms.ToTensor()])

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
    
class ValDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

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





bs = 28
vbs = 8
weight_name = "new_mobnetv2_0.5_iou_{}.pth"
nwork = 8
epochs = 200
wt = "new_mobnetv2_0.5_ba.pth"
start = 211
min_valloss = np.inf
b_iou = 0
weight_b_iou = "new_mobnetv2_0.5_ba.pth"


traindataset = FaceLandmarksDataset(csv_file='aug2_new_train.csv',
                                    root_dir='aug_new_train')

trainloader = DataLoader(traindataset, batch_size=bs,
                                          shuffle=True, num_workers=nwork)

testdataset = ValDataset(csv_file='aug2_new_val.csv',
                                    root_dir='aug_new_val')

testloader = DataLoader(testdataset, batch_size=vbs,
                                          shuffle=False, num_workers=nwork)

model_ft = MobileNetV2(4,width_mult=0.5).cuda()

net = model_ft

import torch.optim as optim

criterion = nn.MSELoss(size_average = True).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)


if wt!=None:
    state = torch.load(wt,map_location=lambda storage, loc: storage.cuda(0))
#    optimizer.load_state_dict(state['optimizer_state_dict'])
    net.load_state_dict(state['model_state_dict'])
    #min_valloss=state['loss']
    b_iou = state['iou']

print("min_val_loss -",min_valloss)
print("b_iou -",b_iou)
train_loss=[]
val_loss=[]

for epoch in range(1,epochs):  # loop over the dataset multiple times
    #t = time.time()
    running_loss = 0.0
    running_loss_val=0.0
    iou_t = 0
    
    net.train()

    for i, data in enumerate(trainloader, 1):
        
        optimizer.zero_grad()
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        
        outputs,iou = net(inputs,labels) 
        iou_loss = torch.tensor(-1.0).cuda()*torch.log(iou+0.000000001)
        loss = criterion(outputs, labels) + 1000*iou_loss
#        loss = criterion(outputs, labels)
#        print(outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iou_t += iou.item()
    print("Epoch- ",epoch+start," Loss- ",running_loss/len(trainloader)," IoU- ",iou_t/len(trainloader))
    train_loss.append(running_loss/len(trainloader))
    #print("Time -",time.time()-t)

    if (epoch+start)%51 == 0:
            print(" Weight saved ",epoch+start)
            torch.save({
            'epoch': epoch+start,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss_val,
            'iou': b_iou,
            },weight_name.format(epoch+start))
    
    
    net.eval()   
    iou_v = 0;
    for i, data in enumerate(testloader, 1):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        
        outputs,iou = net(inputs,labels) 
#            iou_loss = torch.tensor(-1.0).cuda()*torch.log(iou+0.00000000001)
#        loss = criterion(outputs, labels) #+ 1000*iou_loss
#        iou_v += func(outputs.detach().cpu(),labels.detach().cpu())
        iou_loss = torch.tensor(-1.0).cuda()*torch.log(iou+0.000000001)
        loss = criterion(outputs, labels) + 1000*iou_loss
        iou_v += iou.item()
#            outputs = net(inputs)
#            loss = criterion(outputs, labels)
        
#            iou_v += iou.item()
        running_loss_val += loss.item()
    print("  Epoch- ",epoch+start,"Val Loss- ",running_loss_val/len(testloader)," IoU- ",iou_v/len(testloader))
    val_loss.append(running_loss_val/len(testloader))
    if running_loss_val < min_valloss:
        min_valloss = running_loss_val
        print("     Better weight found at ",epoch+start)
        torch.save({
        'epoch': epoch+start,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss_val,
        'iou': b_iou,
        },weight_name.format(epoch+start))
    if iou_v/len(testloader) > b_iou:
        print("     Better iou found at ",epoch+start)
        b_iou = iou_v/len(testloader)
        torch.save({
        'epoch': epoch+start,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss_val,
        'iou': b_iou,
        },weight_b_iou)


print('Finished Training')
