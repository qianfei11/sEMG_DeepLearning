#!/usr/bin/env python3
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

with h5py.File('./DB2/DB2_S1_feature_200_0.h5', 'r') as f:
    featureData = f['featureData'][:]
    featureLabel = f['featureLabel'][:]

featureData = MinMaxScaler().fit_transform(featureData)
xTraining, xTesting, yTraining, yTesting = train_test_split(featureData, featureLabel, test_size=0.2)
print(xTraining.shape)
print(yTraining.shape)

start_time = time.time()
# n_estimators: (default=10) The number of trees in the forest.
# max_features: 寻找最佳分割时要考虑的特征数量
# bootstrap: 默认True，构建树时是否使用bootstrap样本。
RF = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, 
                            min_samples_split=2, min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, 
                            min_impurity_split=None, bootstrap=True, oob_score=True, 
                            n_jobs=1, random_state=None, verbose=0, 
                            warm_start=False, class_weight=None)


RF.fit(xTraining, yTraining)
score = RF.score(xTraining, yTraining)
yPrection = RF.predict(xTesting)
accuracy = metrics.accuracy_score(yTesting, yTesting)

print('RF train accuracy: %.2f%%' % (100 * score))
print('RF test accuracy: %.2f%%' % (100 * accuracy))
print('training took %fs!' % (time.time() - start_time))

