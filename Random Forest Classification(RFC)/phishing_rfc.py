#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 03:08:05 2018

@author: madhurendra
"""
# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('phishing.csv')
dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
arr = [.1,.2,.3,.4,.5, .60,.65,.70,.75,.80,.85,.90,.95]
for i in arr: 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
# Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)

# Predicting the Test set results
    y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    a = cm[0][0]
    b=cm[0][1]
    c=cm[1][1]
    d=cm[1][0]
    print('Total sum :',a+b+c+d)
    print('sum of correct output :', a+c)
    print('Accuracy : ', (a+c)/(a+b+c+d))
