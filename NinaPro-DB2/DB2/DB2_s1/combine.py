#!/usr/bin/env python
import os

SPLIT_LEN = 1024 * 1024 * 10

'''
def splitBigFile(filename):
    foldername = filename.split('.')[0]
    os.mkdir(foldername)
    with open(filename, 'rb') as f:
        data = f.read()
    i = 0
    while SPLIT_LEN * i < len(data):
        with open(foldername + '/' + filename + '.' + str(i), 'wb') as f:
            f.write(data[SPLIT_LEN*i:SPLIT_LEN*i+SPLIT_LEN])
        i += 1

splitBigFile('S1_E1_A1.mat')
splitBigFile('S1_E2_A1.mat')
splitBigFile('S1_E3_A1.mat')
'''

def combineSmallFile(foldername):
    filename = foldername + '.mat'
    length = len(os.listdir(foldername))
    data = ''
    for i in range(length):
        with open(foldername + '/' + filename + '.' + str(i), 'rb') as f:
            data += f.read()
    with open(filename, 'wb') as f:
        f.write(data)

combineSmallFile('S1_E1_A1')
combineSmallFile('S1_E2_A1')
combineSmallFile('S1_E3_A1')

