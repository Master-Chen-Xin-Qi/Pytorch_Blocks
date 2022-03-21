#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : one_hot.py
@Date         : 2022/03/21 22:24:40
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : One hot encoding with scatter
'''
import torch

class_num = 10
batch_size = 4
label = torch.LongTensor(batch_size, 1).random_() % class_num
print(label)
one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
print(one_hot)