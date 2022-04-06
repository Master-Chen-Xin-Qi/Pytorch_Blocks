#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : grad_reverse.py
@Date         : 2022/04/06 21:23:38
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Reverse grad
'''
import torch
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.lambda_value = lambda_value
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_value, None