# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:42:57 2018

@author: alex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    
    def __init__(self):
        super(cnn, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size = 3,padding=0),
                        nn.BatchNorm2d(64,momentum = 1, affine = True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size = 3,padding=0),
                        nn.BatchNorm2d(64,momentum = 1, affine = True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size = 3,padding=0),
                        nn.BatchNorm2d(64,momentum = 1, affine = True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size = 3,padding=0),
                        nn.BatchNorm2d(64,momentum = 1, affine = True),
                        nn.ReLU())
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out
    
    
class RelationNetwork(nn.Module):
    
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        
        def forward(self,x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0),-1)
            out = F.relu(self.fc1(out))
            out = F.sigmoid(self.fc2(out))
            return out
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias.is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        