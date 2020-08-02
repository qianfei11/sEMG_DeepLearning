#!/usr/bin/env python3
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from feature_utils import *  
from scipy import signal
from scipy.fftpack import fft, ifft
import math
import h5py

E1 = scio.loadmat('./DB3/SIA_delsys_16_movements_data/S1_E1.mat')
E2 = scio.loadmat('./DB3/SIA_delsys_16_movements_data/S1_E2.mat')
E3 = scio.loadmat('./DB3/SIA_delsys_16_movements_data/S1_E3.mat')
E4 = scio.loadmat('./DB3/SIA_delsys_16_movements_data/S1_E4.mat')

print(E1.keys())
print(E2.keys())
print(E3.keys())
print(E4.keys())

E1_emg = E1['emg']
E2_emg = E2['emg']
E3_emg = E3['emg']
E4_emg = E4['emg']

E1_label = E1['label']
E2_label = E2['label']
E3_label = E3['label']
E4_label = E4['label']

def getUsefulData(label, emg):
    idx = []
    for i in range(len(label)):
        if label[i] != 0:
            idx.append(i)
    return label[idx, :], emg[idx, :]

label1, emg1 = getUsefulData(E1_label, E1_emg)
label2, emg2 = getUsefulData(E2_label, E2_emg)
label2 += label1[-1, :]
label3, emg3 = getUsefulData(E3_label, E3_emg)
label3 += label2[-1, :]
label4, emg4 = getUsefulData(E4_label, E4_emg)
label4 += label3[-1, :]

emg = np.vstack((emg1, emg2, emg3, emg4))
label = np.vstack((label1, label2, label3, label4))
label -= 1

print(emg.shape)
print(label.shape)
print(label)

# Preprocess
sampleRate = 2000
cutOff = 50

def butterLowpass(data, cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutOff = cutOff / nyq
    b, a = signal.butter(order, normalCutOff, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def butterBandpass(data, fs, freqs, order=8):
    wn1 = 2 * freqs[0] / fs
    wn2 = 2 * freqs[1] / fs
    b, a = signal.butter(order, [wn1, wn2], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

def addGaussianNoise(signalIn, d, SNR=25):
    signalIn_x = np.sum(abs(signalIn) ** 2) / len(signalIn)
    signal_d = np.sum(abs(d) ** 2)
    noise_x = signalIn_x / 10 ** (SNR / 10.)
    noise = np.sqrt(noise_x / signal_d) * d
    noiseSignal = signalIn + noise
    return noiseSignal

def simpleFiltering(y):
    yy = fft(y)
    yf = abs(yy)
    yf1 = yf / (len(y) / 2)
    yf2 = yf1[range(round(len(y) / 2))]
    xf = np.arange(len(yf2))
    plt.plot(xf, yf2)
    plt.show()
    for i in range(len(yy)):
        if i <= 1700 and i >= 300:
            yy[i] = 0
    #y = ifft(yy)
    #yy = fft(y)
    yf = abs(yy)
    yf1 = yf / (len(y) / 2)
    yf2 = yf1[range(round(len(y) / 2))]
    xf = np.arange(len(yf2))
    plt.plot(xf, yf2)
    plt.show()

'''
#y = emg * signal.hanning(512, sym=0)

plt.figure(1)
plt.plot(emg[:, 1])

y = addGaussianNoise(emg[:, 1])
plt.figure(2)
plt.plot(y)

plt.show()

exit()
'''

#for i in range(emg.shape[1]):
#    emg[:, i] = butterLowpass(emg[:, i], cutOff, sampleRate)

'''
# Time Domain Features
featureData = []
featureLabel = []
classes = 16
timeWindow = 200
strideWindow = 200

for i in range(classes):
    index = []
    for j in range(label.shape[0]):
        if label[j, :] == i:
            index.append(j)
    iemg = emg[index, :]
    length = math.floor((iemg.shape[0] - timeWindow) / strideWindow)
    print('[+] class {}, number of sample: {} {}'.format(i, iemg.shape[0], length))
    for j in range(length):
        rms = featureRMS(iemg[strideWindow*j:strideWindow*j+timeWindow, :])
        mav = featureMAV(iemg[strideWindow*j:strideWindow*j+timeWindow, :])
        wl = featureWL(iemg[strideWindow*j:strideWindow*j+timeWindow, :])
        zc = featureZC(iemg[strideWindow*j:strideWindow*j+timeWindow, :])
        ssc = featureSSC(iemg[strideWindow*j:strideWindow*j+timeWindow, :])
        featureStack = np.hstack((rms, mav, wl, zc, ssc))
        featureData.append(featureStack)
        featureLabel.append(i)
featureData = np.array(featureData)

print(featureData.shape)
print(len(featureLabel))

with h5py.File('./DB3/DB3_S1_feature.h5', 'w') as f:
    f.create_dataset('featureData', data=featureData)
    f.create_dataset('featureLabel', data=featureLabel)
'''

imageData = []
imageLabel = []
classes = 16
imageLength = 200
overlapping = 0.6
strideLength = round(imageLength * (1 - overlapping))
print(strideLength)

for i in range(classes):
    index = []
    for j in range(label.shape[0]):
        if label[j, :] == i:
            index.append(j)
    iemg = emg[index, :]
    length = math.floor((iemg.shape[0] - imageLength) / strideLength)
    print('[+] class {}, number of sample: {} {}'.format(i, iemg.shape[0], length))
    for j in range(length):
        subImage = iemg[strideLength*j:strideLength*j+imageLength, :]
        imageData.append(subImage)
        imageLabel.append(i)

imageData = np.array(imageData)
print(imageData.shape)
print(len(imageLabel))

with h5py.File('./DB3/data.h5', 'w') as f:
    f.create_dataset('imageData', data=imageData)
    f.create_dataset('imageLabel', data=imageLabel)

