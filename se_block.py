#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : se_block.py
@Date         : 2022/03/20 21:06:03
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Squeeze and Excitation block
'''
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y