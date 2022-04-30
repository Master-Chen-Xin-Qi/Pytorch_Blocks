#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : plot_fft.py
@Date         : 2022/04/30 12:07:07
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : plot fft spectrum
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N = 600
# sample spacing, 1/fs
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0 * np.pi * x + np.sin(30 * 2.0 * np.pi * x)) 
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()