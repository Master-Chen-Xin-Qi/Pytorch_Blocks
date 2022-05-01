#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : scipy_stft.py
@Date         : 2022/05/01 15:30:22
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : The example of scipy stft
'''

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

'''
    args:
        x: the input signal
        fs: the sampling frequency
        npserseg: the number of samples in each segment
        noverlap: the number of samples overlap between segments(optional, default is nperseg//2)
        nfft: the number of samples in the FFT computation(optional, default is npserseg)
    return:
        f: the frequency vector
        t: the time vector
        Zxx: the STFT matrix, remember to use np.abs(Zxx) to get the magnitude
'''

if __name__ == "__main__":
    # 官方样例，分析调频信号 
    
    rng = np.random.default_rng()

    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)  # 10s的数据
    mod = 500*np.cos(2*np.pi*0.25*time)
    carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    noise = rng.normal(scale=np.sqrt(noise_power),
                    size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + noise

    f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    # 实际处理一段音频信号，是长度为30s的线性扫频信号，从1Hz到96kHz
    
    # fs, data = wavfile.read('C:/Users/Master/Desktop/audiocheck.net_hdsweep_1Hz_96000Hz_-3dBFS_30s (1)(1)(1).wav', 'r')
    
    # f, t, Zxx = signal.stft(data, fs, nperseg=192000)
    # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.707, shading='gouraud')
    # plt.colorbar()
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()