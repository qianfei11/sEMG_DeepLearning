#!/usr/bin/env python3
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from feature_utils import *
import math
from sklearn.preprocessing import MinMaxScaler
import h5py

E1 = scio.loadmat('./DB2/DB2_s1/S1_E1_A1.mat')
E2 = scio.loadmat('./DB2/DB2_s1/S1_E2_A1.mat')
E3 = scio.loadmat('./DB2/DB2_s1/S1_E3_A1.mat')

print(E1.keys())
print(E2.keys())
print(E3.keys())

# 肌电信号
E1_emg = E1['emg']
E2_emg = E2['emg']
E3_emg = E3['emg']

# 刺激反应
E1_label = E1['restimulus']
E2_label = E2['restimulus']
E3_label = E3['restimulus']

def getUsefulData(label, emg):
    idx = []
    for i in range(len(label)):
        if label[i] != 0:
            idx.append(i)
    return label[idx, :], emg[idx, :]

label1, emg1 = getUsefulData(E1_label, E1_emg)
label2, emg2 = getUsefulData(E2_label, E2_emg)
label3, emg3 = getUsefulData(E3_label, E3_emg)

emg = np.vstack((emg1, emg2, emg3))
label = np.vstack((label1, label2, label3))
label -= 1

print(emg.shape)
print(label.shape)
print(label)

#plt.plot(E1_emg[:200000, 5] * 20000)
#plt.plot(E1_label[0:200000])
#plt.show()

featureData = []
featureLabel = []
classes = 49 # 肌电信号种类
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

emg *= 20000
imageData = []
imageLabel = []
imageLength = 200

for i in range(classes):
    index = [];
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

with h5py.File('./DB2/DB2_S1_feature_200_0.h5', 'w') as f:
    f.create_dataset('featureData', data=featureData)
    f.create_dataset('featureLabel', data=featureLabel)

with h5py.File('./DB2/DB2_S1_image_200_0.h5', 'w') as f:
    f.create_dataset('imageData', data=imageData)
    f.create_dataset('imageLabel', data=imageLabel)

