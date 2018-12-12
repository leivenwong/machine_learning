import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from learning_settings import Settings
import learning_functions as fuc
import data_analyze
import linear_model

import sys
sys.path.append('D:\\python_project\\statistics')
from statistics_functions import compute_r

#initiate settings
ai_settings = Settings()

#read raw data
target_raw = data_analyze.target
target_raw = pd.DataFrame(target_raw)

#here are some made-up numbers to start with
target = list(linear_model.yTest)
prediction = linear_model.testPredictions

error = []
for i in range(len(target)):
    error.append(target[i] - prediction[i])

#print the errors
print("Errors ",)
print(error)

#calculate the squared errors and absolute value of errors
squaredError = []
absError = []
for val in error:
    squaredError.append(val*val)
    absError.append(abs(val))

#print squared errors and absolute value of errors
print("Squared Error")
print(squaredError)

print("Absolute Value of Error")
print(absError)

#calculate and print mean squared error MSE
print("MSE = ", sum(squaredError)/len(squaredError))

from math import sqrt
#calculate and print square root of MSE (RMSE)
print("RMSE = ", sqrt(sum(squaredError)/len(squaredError)))

#calculate and print mean absolute error MAE
print("MAE = ", sum(absError)/len(absError))

#compare MSE to target variance
targetDeviation = []
targetMean = sum(target)/len(target)
for val in target:
    targetDeviation.append((val - targetMean)*(val - targetMean))

#print the target variance
print("Target Variance = ", sum(targetDeviation)/len(targetDeviation))

#print the the target standard deviation (square root of variance)
print("Target Standard Deviation = ", sqrt(sum(targetDeviation)/len(targetDeviation)))
