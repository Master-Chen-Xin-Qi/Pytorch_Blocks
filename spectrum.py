# import the libraries
import matplotlib.pyplot as plot
import numpy as np

# Define the list of frequencies
frequencies = np.arange(5,105,5)


# Sampling Frequency
samplingFrequency = 400

# Create two ndarrays
s1 = np.empty([0]) # For samples
s2 = np.empty([0]) # For signal

# Start Value of the sample
start = 1

# Stop Value of the sample
stop = samplingFrequency+1

for frequency in frequencies:
    sub1 = np.arange(start, stop, 1)

    # Signal - Sine wave with varying frequency + Noise
    sub2 = np.sin(2*np.pi*sub1*frequency*1/samplingFrequency)+np.random.randn(len(sub1))
    s1 = np.append(s1, sub1)
    s2 = np.append(s2, sub2)
    start = stop+1
    stop = start+samplingFrequency

 
# Plot the signal
plot.subplot(211)
plot.plot(s1,s2)
plot.xlabel('Sample')
plot.ylabel('Amplitude')

# Plot the spectrogram
#################
# Fs: 采样频率
# NFFT: FFT的点数，默认是256，点数越小，时间分辨率越精细，点数越大，频率分辨率越精细
# noverlap: 两个FFT之间的重叠点数，默认是128，当数据点比较少的时候可以将其调大，增加重叠部分
# 例子：假如300个点，采样率50Hz，则一共是6s的数据。设置Fs=50，NFFT=50，noverlap=0，则每次取50个点，
# 无重叠，横轴正好是6秒
#################
plot.subplot(212)
powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(s2, Fs=samplingFrequency)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.show() 

# template
NFFT = 2**7
FS = 100
spectrum, freqs, time, imageAxis = plot.specgram(single_data, 
                            NFFT=NFFT, 
                            window=np.hanning(M=NFFT),
                            Fs=FS, 
                            noverlap=NFFT*0.8,
                            sides='onesided',
                            mode='psd',
                            scale_by_freq=True,
                            # cmap="plasma",  #color
                            detrend='linear',
                            xextent=None)