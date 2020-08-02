#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch.nn.functional as F

EPOCH = 100
BATCH_SIZE = 512
LR = 0.05

with h5py.File('./data/data.h5', 'r') as f:
    imageData = f['imageData'][:]
    #print(imageData.shape)
    imageLabel = f['imageLabel'][:]
    #print(imageLabel.shape)

# 随机打乱数据和标签
N = imageData.shape[0]
index = np.random.permutation(N)
data = imageData[index, :, :]
data = data.reshape(-1, 10, 12)
#print(data.shape)
label = imageLabel[index]

# 划分数据集
N = data.shape[0]
num_train = round(N * 0.8)
xTraining = data[:num_train, :, :]
yTraining = label[:num_train]
xTesting = data[num_train:, :, :]
yTesting = label[num_train:]

xTraining, yTraining = torch.from_numpy(xTraining.astype(np.float32)), torch.from_numpy(yTraining.astype(np.long))
xTesting, yTesting = torch.from_numpy(xTesting.astype(np.float32)), torch.from_numpy(yTesting.astype(np.long))

#print('xTraining shape:', xTraining.shape)
#print('yTraining shape:', yTraining.shape)
#print('xTesting shape:', xTesting.shape)
#print('yTesting shape:', yTesting.shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 10 * 12
            nn.Conv1d(
                in_channels=10,
                out_channels=32,
                kernel_size=6,
                stride=1,
                padding=3, 
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # -> 64 * 6
        )
        self.conv3 = nn.Sequential( # 64 * 6
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=2,
                stride=1,
                padding=1, 
            ), # -> 128 * 6
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # -> 128 * 3
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(768, 384), 
            nn.BatchNorm1d(384, momentum=0.5), 
            nn.ReLU(), 
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(384, 128), 
            nn.BatchNorm1d(128, momentum=0.5), 
            nn.ReLU(), 
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(128, 52), 
            nn.BatchNorm1d(52, momentum=0.5), 
            nn.Softmax(dim=1), 
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1(x)
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x1 = self.conv3(x1)
        x2 = self.conv3(x2)
        x1 = x1.view(x1.size(0), -1) # (batch, 128 * 5)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        #print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        #print(output.size())
        return output

if __name__ == '__main__':
    cnn = CNN()
    print(cnn)
    cnn.load_state_dict(torch.load('./paramsBest-2.pkl'))
    torchDataset = Data.TensorDataset(xTraining, yTraining)
    trainLoader = Data.DataLoader(dataset=torchDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    #optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
    lossFunc = nn.CrossEntropyLoss()
    bestAccuracy = 0
    idx = 0
    for epoch in range(EPOCH):
        for step, (bX, bY) in enumerate(trainLoader):
            output = cnn(bX)
            #print(bY.size())
            loss = lossFunc(output, bY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 0:
                testOutput = cnn(xTesting)
                predY = torch.max(testOutput, 1)[1].data.numpy()
                accuracy = float((predY == yTesting.data.numpy()).astype(int).sum()) / float(yTesting.size(0))
                print('[+] Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, accuracy))
                if accuracy > bestAccuracy:
                    if idx != 0:
                        torch.save(cnn.state_dict(), 'params-{}.pkl'.format(idx))
                    bestAccuracy = accuracy
                    idx += 1
                    print('[-] Best Accuracy: {}'.format(bestAccuracy))
    trainOutput = cnn(xTraining)
    loss = lossFunc(trainOutput, yTraining)
    predY = torch.max(trainOutput, 1)[1].data.numpy()
    accuracy = float((predY == yTraining.data.numpy()).astype(int).sum()) / float(yTraining.size(0))
    print('[+] Train Accuracy: {}, Train Loss: {}'.format(accuracy, loss))
    testOutput = cnn(xTesting)
    loss = lossFunc(testOutput, yTesting)
    predY = torch.max(testOutput, 1)[1].data.numpy()
    accuracy = float((predY == yTesting.data.numpy()).astype(int).sum()) / float(yTesting.size(0))
    print('[+] Test Accuracy: {}, Test Loss: {}'.format(accuracy, loss))

