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
import csv
import re


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
        self.features = [conv_bn(1, input_channel, 2)]
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
        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(),transforms.ToTensor()])

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread("training/"+img_name)

        name = self.landmarks_frame.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return [image,name]



nwork = 4
wt = 'mobnetv2_0.75_gray_smoothl1_130.pth'

testdataset = FaceLandmarksDataset(csv_file='new_train2.csv',
                                    root_dir='')

testloader = DataLoader(testdataset, batch_size=1,
                                          shuffle=False, num_workers=nwork)

model_ft = MobileNetV2(4,width_mult=0.75).cuda()

net = model_ft

import torch.optim as optim


min_valloss = np.inf

if wt!=None:
    state = torch.load(wt,map_location=lambda storage, loc: storage.cuda(0))
#    optimizer.load_state_dict(state['optimizer_state_dict'])
    net.load_state_dict(state['model_state_dict'])
    min_valloss=state['loss']
    print(min_valloss)
ls=[]
net.eval()     
for i, data in enumerate(testloader, 1):
    inputs, name = data
    inputs = Variable(inputs.cuda())
    outputs = net(inputs)
    ls.append([name,outputs.detach().cpu().numpy()])

ans_file = "temp2train_0.75.csv" 
final_sub = "new_train2_0.75.csv"
	
with open(ans_file,'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(ls)
	

ff =  open(final_sub,'w') 
with open(ans_file) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for columns in csv_reader:
		text = columns[0]
		found = re.search("'(.+?)',",text).group(1)
		list_str = columns[1]
		found2 = re.search("\[\[(.+?)\]\]",list_str).group(1)		
		final = re.sub(' +', ' ', found2)
		final = final.strip().replace(" ",",")
		print(found,",",final)

		ff.write(found+","+final+"\n")
ff.close()


