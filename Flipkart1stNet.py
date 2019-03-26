#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:47:35 2019

@author: shourya
"""


import torch
from torchvision import models
import torchvision.transforms as transforms
import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import torch.nn as nn



class FDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread("train_imgs/"+img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        y = np.asarray([landmarks[0],landmarks[1],landmarks[2],landmarks[3]])
        image = self.transform(image)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        return [image,y]

dataset = FDataset(csv_file='training.csv',
                                    root_dir='')


trainloader = DataLoader(dataset, batch_size=4,
                                          shuffle=True, num_workers=2)



model_ft = models.resnet18(pretrained=False)
#num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Linear(64512, 2000),nn.Linear(2000, 512),nn.Linear(512, 4))

net = model_ft.cuda()

import torch.optim as optim

criterion = nn.MSELoss(size_average = True).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)



for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #if i==3:
         #   break
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    print("Epoch- ",epoch," Loss- ",running_loss/len(trainloader))
print('Finished Training')



