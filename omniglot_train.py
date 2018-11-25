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
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random



def main():
    print ("init data folders")
    metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()
    


