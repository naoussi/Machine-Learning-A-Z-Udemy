#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 07:06:12 2018

@author: comp-guru
"""

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Splitting Dataset into Training And Validation Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

#Using Regressor To Predict Test Set and Calculate Accuracy
y_pred = linear_regressor.predict(X_test)

#visualising the Training Set Results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,  linear_regressor.predict(X_train), color='blue')
plt.title("Plot of Salary Vs Years Of Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#visualising the Test Set Results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,  linear_regressor.predict(X_train), color='blue')
plt.title("Plot of Salary Vs Years Of Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
