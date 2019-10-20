#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:38:05 2018

@author: madhurendra
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset

dataset = pd.read_csv('phishing.csv')
dataset = dataset.drop('id', 1) #removing unwanted column
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Splitting the dataset into the Training set and Test set
arr = [.1,.2,.3,.4,.5, .60,.65,.70,.75,.80,.85,.90,.95]
for i in arr: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = float(i), random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print('mkp')
    classifier = Sequential()

    classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu', input_dim = 30))

# Adding the second hidden layer
    classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

# Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 30, nb_epoch = 100)


# Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    a = cm[0][1]
    b=cm[0][2]
    c=cm[2][2]
    d=cm[2][1]
    print('Total sum :',a+b+c+d)
    print('sum of correct output :', a+c)
    print('Accuracy : ', (a+c)/(a+b+c+d))
