#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : warmup.py
@Date         : 2022/03/20 20:34:28
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : warmup for learning rate
'''
from math import cos, pi
import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet18


def adjust_learning_rate(optimizer, lr, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=10):
    if current_epoch < warmup_epoch:
        lr += (lr_max - lr) / (warmup_epoch - current_epoch)
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 
model = resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
 
lr_max=0.1
lr_min=0.00001
max_epoch=50
lrs=[]
for epoch in range(0, max_epoch):
    print(optimizer.param_groups[0]['lr'])
    lrs.append(optimizer.param_groups[0]['lr'])
    adjust_learning_rate(optimizer=optimizer, lr=optimizer.param_groups[0]['lr'], current_epoch=epoch,max_epoch=max_epoch,
                         lr_min=lr_min,lr_max=lr_max,warmup_epoch=10)
    optimizer.step()
 
plt.plot(lrs)
plt.show()