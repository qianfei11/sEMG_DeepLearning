#!/usr/bin/env python3
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time

with h5py.File('./DB3/DB3_S1_feature.h5', 'r') as f:
    featureData = f['featureData'][:]
    featureLabel = f['featureLabel'][:]

featureData = MinMaxScaler().fit_transform(featureData)
xTraining, xTesting, yTraining, yTesting = train_test_split(featureData, featureLabel, test_size=0.1)

print(featureData.shape)

start_time = time.time()
RF = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, 
                            min_samples_split=2, min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, 
                            min_impurity_split=None, bootstrap=True, oob_score=True, 
                            n_jobs=1, random_state=None, verbose=0, 
                            warm_start=False, class_weight=None)

RF.fit(xTraining, yTraining)
score = RF.score(xTraining, yTraining)
yPrediction = RF.predict(xTesting)
accuracy = metrics.accuracy_score(yTesting, yPrediction)

print('RF train accuracy: %.2f%%' % (100 * score))
print('RF test accuracy: %.2f%%' % (100 * accuracy))
print('training took %fs!' % (time.time() - start_time))

SVM = SVC(C=2, kernel='rbf', degree=3, gamma=2)
SVM.fit(xTraining, yTraining)
score = SVM.score(xTraining, yTraining)
predict_SVM = SVM.predict(xTesting)
accuracy_SVM = metrics.accuracy_score(yTesting, predict_SVM)
print(score, accuracy_SVM)

