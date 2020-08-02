#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import h5py
import numpy as np

EPOCH = 100
BATCH_SIZE = 512
LR = 0.05

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
data = np.expand_dims(data, axis=3)
#print(data.shape)
data = data.reshape(-1, 1, 200, 6)
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
        f1 = [20, 16, 12, 8]
        f2 = [3, 4, 5, 6]
        self.conv1 = []
        for i in range(4):
            self.conv1.append(nn.Sequential( # 1 * 200 * 6
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(f1[i], 3),
                    stride=(1, 1),
                ), # -> 32 * 181 * 4
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(20, 1)), # -> 32 * 20
            ))
        self.conv2 = []
        for i in range(4):
            self.conv2.append(nn.Sequential( # 6 * 200
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=(f2[i], 1),
                    stride=(1, 1),
                ), # -> 32 * 200
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(9-2-i, 1)), # -> 32 * 20
            ))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(1024, 128), 
            nn.BatchNorm1d(128, momentum=0.5), 
            nn.ReLU(), 
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(128, 16), 
            nn.BatchNorm1d(16, momentum=0.5), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = [x for i in range(4)]
        for i in range(4):
            x[i] = self.conv1[i](x[i])
            #print(x[i].shape)
            x[i] = self.conv2[i](x[i])
            #print(x[i].shape)
            x[i] = x[i].view(x[i].size(0), -1)
            #print(x[i].shape)
        x = torch.cat((x[0], x[1], x[2], x[3]), dim=1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output

if __name__ == '__main__':
    cnn = CNN()
    #print(cnn)
    #cnn.load_state_dict(torch.load('./paramsBest-0.pkl'))
    torchDataset = Data.TensorDataset(xTraining, yTraining)
    trainLoader = Data.DataLoader(dataset=torchDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    #optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.8)
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

