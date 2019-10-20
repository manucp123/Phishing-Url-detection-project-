#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 2`:28:05 2018

@author: madhurendra
"""
# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('phishing.csv')
dataset = dataset.drop('id', 1) #removing unwanted column
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values

# Importing  the important libraries
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
arr = [.1,.2,.3,.4,.5, .60,.65,.70,.75,.80,.85,.90,.95]
for i in arr: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)

# Predicting the Test set results
    y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    a = cm[0][0]
    b=cm[0][1]
    c=cm[1][1]
    d=cm[1][0]
    print('Total sum :',a+b+c+d)
    print('sum of correct output :', a+c)
    print('Accuracy : ', (a+c)/(a+b+c+d))
