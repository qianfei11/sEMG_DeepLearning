#!/usr/bin/env python3
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from feature_utils import *  
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

#plt.plot(emg)
#plt.plot(label*0.0001)
#plt.show()

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

imageData = []
imageLabel = []
imageLength = 200

for i in range(classes):
    index = []
    for j in range(label.shape[0]):
        if label[j, :] == i:
            index.append(j)
    iemg = emg[index, :]
    length = math.floor((iemg.shape[0] - imageLength) / imageLength)
    print('[+] class {}, number of sample: {} {}'.format(i, iemg.shape[0], length))
    for j in range(length):
        subImage = iemg[imageLength*j:imageLength*(j+1), :]
        imageData.append(subImage)
        imageLabel.append(i)

imageData = np.array(imageData)
print(imageData.shape)
print(len(imageLabel))

with h5py.File('./DB3/DB3_S1_feature.h5', 'w') as f:
    f.create_dataset('featureData', data=featureData)
    f.create_dataset('featureLabel', data=featureLabel)

with h5py.File('./DB3/DB3_S1_image.h5', 'w') as f:
    f.create_dataset('imageData', data=imageData)
    f.create_dataset('imageLabel', data=imageLabel)

