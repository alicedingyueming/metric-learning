# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:00:25 2018

@author: alex
"""
import os
import random


def omniglot_character_folders():
    data_folder = 'C:\\Users\\alex\\metric-learning\\datas\\omniglot_resized\\'

    character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)

    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]

    return metatrain_character_folders,metaval_character_folders