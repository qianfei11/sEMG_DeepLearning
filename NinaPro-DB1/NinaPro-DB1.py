#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, ZeroPadding2D, ZeroPadding1D, Dropout, Activation, Flatten, Conv2D, Conv1D, AveragePooling2D, AveragePooling1D, MaxPooling2D, MaxPooling1D, concatenate, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt

with h5py.File('./data/DB1_S1_image.h5.bak', 'r') as f:
    imageData = f['imageData'][:]
    imageLabel = f['imageLabel'][:]

# 随机打乱数据和标签
N = imageData.shape[0]
index = np.random.permutation(N)
data  = imageData[index, :, :]
label = imageLabel[index]

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# 对数据升维 标签one-hot
data = np.expand_dims(data, axis=2)
label = convert_to_one_hot(label, 52).T

# 划分数据集
N = data.shape[0]
num_train = round(N * 0.8)
X_train = data[0:num_train, :, :]
Y_train = label[0:num_train, :]
X_test  = data[num_train:N, :, :]
Y_test  = label[num_train:N, :]

print('X_train shape:', str(X_train.shape))
print('Y_train shape:', str(Y_train.shape))
print('X_test shape:', str(X_test.shape))
print('Y_test shape:', str(Y_test.shape))

# 写一个LossHistory类 保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc='upper right')
        plt.show()

def CNN(input_shape=(12, 1, 10), classes=52): 
    X_input = Input(input_shape)
    X = Conv2D(filters=32, kernel_size=(6, 1), strides=(1, 1), padding='same', activation='relu', name='conv1')(X_input)
    X = Conv2D(filters=64, kernel_size=(4, 1), strides=(1, 1), padding='same', activation='relu', name='conv2')(X)
    X = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool1')(X)
    X = Conv2D(filters=128, kernel_size=(2, 1), strides=(1, 1), padding='same', activation='relu', name='conv3')(X)
    X = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='pool2')(X)
    X = Flatten(name='flatten')(X)
    X = Dropout(0.5)(X)
    X = Dense(128, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc2')(X)
    model = Model(inputs=X_input, outputs=X, name='CNN')
    return model

model = CNN(input_shape=(12, 1, 10), classes=52)
model.summary()

import time
start = time.time()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = LossHistory() # 创建一个history实例

model.fit(X_train, Y_train, epochs=200, batch_size=64, verbose=1, 
            validation_data=(X_test, Y_test), callbacks=[history])

preds_train = model.evaluate(X_train, Y_train)
print('Train Loss =', str(preds_train[0]))
print('Train Accuracy =', str(preds_train[1]))

preds_test  = model.evaluate(X_test, Y_test)
print('Test Loss =', str(preds_test[0]))
print('Test Accuracy =', str(preds_test[1]))

end = time.time()
print('time:', end - start)

history.loss_plot('epoch')

