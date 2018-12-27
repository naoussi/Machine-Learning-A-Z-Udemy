#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 05:23:46 2018

@author: comp-guru
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Taking Care Of Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0 )
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding Categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

one_hot_encoder_X = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder_X.fit_transform(X).toarray()

#Splitting Dataset into Training And Validation Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)