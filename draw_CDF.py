#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : draw_CDF.py
@Date         : 2022/05/04 20:18:48
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Draw cumulative distribution function of input data
'''

import seaborn
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(50)
seaborn.ecdfplot(data)
plt.xlabel('Data')
plt.show()