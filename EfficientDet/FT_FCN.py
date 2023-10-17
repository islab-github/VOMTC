# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 18:36:29 2022

@author: Glen Kim
"""

import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device2 = torch.device("cuda:2")
device3 = torch.device("cuda:3")


class FT_FCN(nn.Module):
    def __init__(self):
        super(FT_FCN, self).__init__()
        self.split_size = 2
        self.linear1 = nn.Linear(400, 300).to(device2)   
        self.bn1 = nn.BatchNorm1d(300).to(device2)  #.to(device2)    
        
        self.linear2 = nn.Linear(300, 200 ).to(device2)
        self.bn2 = nn.BatchNorm1d(200).to(device2)  #.to(device3)
        self.relu2 = nn.ReLU()
        
        self.linear3 = nn.Linear(200, 300).to(device2)
        self.bn3 = nn.BatchNorm1d(300).to(device2) #.to(device1)
        self.relu3 = nn.ReLU()
        
        
        self.linear4 = nn.Linear(300, 400).to(device2)
        
            
        self.sigmoid = nn.Sigmoid().to(device2)  #.to(device1)
        
    def forward(self, tensors):
        
        
        
        
        a = self.linear1(tensors)
        aa = self.bn1(a) #a.to(device2)
        
       
        
        b = self.linear2(aa) #aa.detach().cpu()
        bb = self.bn2(b)
        bbb = self.relu2(bb)
        
        c = self.linear3(bbb) #bbb.detach().cpu()
        cc = self.bn3(c)
        ccc = self.relu3(cc)
        
        out = self.linear4(ccc) #ccc.detach().cpu()
        
        out = self.sigmoid(out) #out.to(device1)
        return out
    
    
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
            
    def init_FT_FCN(self, path):
        state_dict = torch.load(path)
        #try:
        #model = self.load_state_dict(state_dict, strict=False)
        return state_dict
        #except RuntimeError as e:
        #    print("Ignoring " + str(e) + " ")

def build_DNN_refiner():
    
        
    
    return FT_FCN()
#