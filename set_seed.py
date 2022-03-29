#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : set_seed.py
@Date         : 2022/03/29 21:30:26
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Set random seed
'''

import torch
import numpy as np

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False