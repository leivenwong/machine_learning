import urllib.request
from sklearn import datasets, preprocessing, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from learning_settings import Settings
import learning_functions as fuc
import data_analyze


#read data from uci data repository
#initiate settings
ai_settings = Settings()

#read raw data
target_raw = data_analyze.target

#here are some made-up numbers to start with
data = target_raw[['rsi', 'profit_per_roll', 'macd']]
prediction = target_raw['profit_per']

#arrange data into list for labels and list of lists for attributes
xList = data
labels = prediction

#compute train and test data critical value
cut = int(len(labels) / 3)

#divide attribute matrix and label vector into training(2/3 of data) and test sets (1/3 of data)
xListTest = xList[cut:]
xListTrain = xList[:cut]
labelsTest = labels[cut:]
labelsTrain = labels[:cut]

xTrain = xListTrain
xTest = xListTest
yTrain = labelsTrain
yTest = labelsTest
print(yTest)

#train linear regression model
model = linear_model.LinearRegression(fit_intercept=True, normalize=False,
    copy_X=True, n_jobs=1)
model.fit(xTrain, yTrain)


#generate predictions on in-sample error
trainingPredictions = model.predict(xListTrain)
print("Some values predicted by model", trainingPredictions[0:5], trainingPredictions[-6:-1])

print("model parameter: " + str(model.get_params()))
print("model score: " + str(model.score(xListTest, labelsTest)))
print("model coef: " + str(model.coef_))
print("model intercept: " + str(model.intercept_))

#generate predictions on out-of-sample data
testPredictions = model.predict(xTest)
print("testpredictions:")
print(testPredictions)