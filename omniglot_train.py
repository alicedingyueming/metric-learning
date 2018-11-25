# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:50:30 2018

@author: alex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random

import Modules


def run(dataloader, model, mode, writer, learning_rate, momentum, epoch_id):
    pass

def main():
    useCUDA = False
    LEARNING_RATE = 0.001
    
    print ("init data folders")
    metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()
#    print("train",metatrain_character_folders)
#    print("test",metatest_character_folders)
    feature_encoder = Modules.cnn()
    relation_network = Modules.RelationNetwork()
    feature_encoder.apply(Modules.weights_init)
    relation_network.apply(Modules.weights_init)
    
    if useCUDA:
        feature_encoder.cuda()
        relation_network.cuda()
        
    feature_optim = optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    relation_network_optim = optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)
    
    print("training...")
    
    
    

if __name__ == "__main__":
    main()
