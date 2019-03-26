from __future__ import print_function, division, absolute_import
from torch.autograd import Variable
import numpy as np


import torch
import torch.nn as nn
import time
import torchvision.transforms as transforms
import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F




class Inception3(nn.Module):

    def __init__(self, num_classes=4, aux_logits=True, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        #print(x.size())
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        #print(x.size())
        # N x 768 x 5 x 5
        x = self.conv0(x)
        #print(x.size())
        # N x 128 x 5 x 5
        x = self.conv1(x)
        #print(x.size())
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        #print(x.size())
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        #print(x.size())
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



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



bs = 18
vbs = 2
weight_name = "inceptionv3_best.pth"
nwork = 4
epochs = 200
wt = "inceptionv3_best.pth"
start = 48

traindataset = FaceLandmarksDataset(csv_file='new_train.csv',
                                    root_dir='/media/guest/84B0DF27B0DF1F0C/aaa/oldie/FlipkartGridStage2DataSetImages/images/')

trainloader = DataLoader(traindataset, batch_size=bs,
                                          shuffle=True, num_workers=nwork)

testdataset = ValDataset(csv_file='new_val.csv',
                                    root_dir='/media/guest/84B0DF27B0DF1F0C/aaa/oldie/FlipkartGridStage2DataSetImages/images/')

testloader = DataLoader(testdataset, batch_size=vbs,
                                          shuffle=False, num_workers=nwork)

model_ft = Inception3().cuda()

net = model_ft

import torch.optim as optim

criterion = nn.SmoothL1Loss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=4)

min_valloss = np.inf

if wt!=None:
    state = torch.load(wt,map_location=lambda storage, loc: storage.cuda(0))
#    optimizer.load_state_dict(state['optimizer_state_dict'])
    net.load_state_dict(state['model_state_dict'])
    min_valloss=state['loss']

print(min_valloss)
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
        loss_T = criterion(outputs[0], labels)
        loss_T2 = criterion(outputs[1], labels)
        loss = loss_T + loss_T2
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

#    if (epoch+start)%51 == 0:
#            print(" Weight saved ",epoch+start)
#            torch.save({
#            'epoch': epoch+start,
#            'model_state_dict': net.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': avg_train_loss
#            },weight_name)
#    
#    
    net.eval()   

    if epoch==1 or epoch==2:
        t = time.time()
        
    for i, data in enumerate(testloader, 1):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        outputs = net(inputs) 
       
        loss = criterion(outputs, labels)
        #loss_T2 = criterion(outputs[1], labels)
#        loss = loss_T #+ loss_T2
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
        },weight_name)
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

    
    