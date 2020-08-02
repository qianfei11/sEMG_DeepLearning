#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import h5py
import numpy as np

EPOCH = 100
BATCH_SIZE = 64
LR = 1e-2

with h5py.File('./DB3/DB3_S1_image.h5', 'r') as f:
    imageData = f['imageData'][:]
    #print(imageData.shape)
    imageData = imageData * 2000
    imageLabel = f['imageLabel'][:]
    #print(imageLabel.shape)

# 随机打乱数据和标签
N = imageData.shape[0]
index = np.random.permutation(N)
data = imageData[index, :, :]
data = data.reshape(-1, 6, 200)
#print(data.shape)
label = imageLabel[index]

# 划分数据集
N = data.shape[0]
num_train = round(N * 0.8)
xTraining = data[0:num_train, :, :]
#yTraining = label[0:num_train, :]
yTraining = label[0:num_train]
xTesting = data[num_train:N, :, :]
#yTesting = label[num_train:N, :]
yTesting = label[num_train:N]

xTraining, yTraining = torch.from_numpy(xTraining), torch.from_numpy(yTraining)
xTesting, yTesting = torch.from_numpy(xTesting), torch.from_numpy(yTesting)

#print('xTraining shape:', xTraining.shape)
#print('yTraining shape:', yTraining.shape)
#print('xTesting shape:', xTesting.shape)
#print('yTesting shape:', yTesting.shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 6 * 200
            nn.Conv1d(
                in_channels=6,
                out_channels=32,
                kernel_size=20,
                stride=1,
                padding=10,
            ), # -> 32 * 200
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=10), # -> 32 * 20
        )
        self.conv2 = nn.Sequential( # 32 * 20
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=6,
                stride=1,
                padding=3,
            ), # -> 64 * 20
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # -> 64 * 10
        )
        self.conv3 = nn.Sequential( # 64 * 10
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ), # -> 128 * 10
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # -> 128 * 5
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(1280, 640), 
            nn.BatchNorm1d(640, momentum=0.5), 
            nn.ReLU(), 
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(640, 128), 
            nn.BatchNorm1d(128, momentum=0.5), 
            nn.ReLU(), 
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(128, 16), 
            nn.BatchNorm1d(16, momentum=0.5), 
            nn.Softmax(dim=1), 
        )

    def forward(self, x):
        #print(x.size())
        x1 = self.conv1(x)
        x2 = self.conv1(x)
        #print(x.size())
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        #print(x.size())
        x1 = self.conv3(x1) # (batch, 128, 5)
        x2 = self.conv3(x2)
        x1 = x1.view(x.size(0), -1) # (batch, 128 * 5)
        x2 = x2.view(x.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        #print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

if __name__ == '__main__':
    cnn = CNN()
    print(cnn)
    #cnn.load_state_dict(torch.load('./paramsBest-0.pkl'))
    torchDataset = Data.TensorDataset(xTraining, yTraining)
    trainLoader = Data.DataLoader(dataset=torchDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
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

