import urllib.request
from sklearn import datasets, preprocessing, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl
from math import sqrt
import pymysql
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from learning_settings import Settings
import learning_functions as fuc
import data_analyze


def xattrSelect(x, idxSet):
    #takes X matrix as list of list and returns subset containing columns in idxSet
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return(xOut)

#initiate settings
ai_settings = Settings()

#read raw data
target_raw = data_analyze.target

#here are some made-up numbers to start with
data = target_raw[['macd', 'rsi', 'ema', 'profit_per_roll']].values.tolist()
prediction = list(target_raw['profit_per'])

#arrange data into list for labels and list of lists for attributes
xList = data
labels = prediction
names = ['macd', 'rsi', 'ema', 'profit_per_roll', 'profit_per']
print(xList)
print(labels)

#compute train and test data critical value
cut = int(len(labels) / 3)

#divide attribute matrix and label vector into training(2/3 of data) and test sets (1/3 of data)
indices = range(len(xList))
xListTest = xList[cut:]
xListTrain = xList[:cut]
labelsTest = labels[cut:]
labelsTrain = labels[:cut]

xTrain = numpy.array(xListTrain)
yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)

alphaList = [0.1**i for i in range(10)]

rmsError = []
for alph in alphaList:
    model = linear_model.Ridge(alpha=alph)
    model.fit(xTrain, yTrain)
    rmsError.append(numpy.linalg.norm((yTest-model.predict(xTest)), 2)/sqrt(len(yTest)))

print("RMS Error                 alpha")
for i in range(len(rmsError)):
    print(rmsError[i], alphaList[i])

#plot curve of out-of-sample error versus alpha
x = range(len(rmsError))
plt.plot(x, rmsError, 'k')
plt.xlabel('-log(alpha)')
plt.ylabel('Error (RMS)')
plt.show()

#Plot histogram of out of sample errors for best alpha value and scatter plot of actual versus predicted
#Identify index corresponding to min value, retrain with the corresponding value of alpha
#Use resulting model to predict against out of sample data.  Plot errors (aka residuals)
indexBest = rmsError.index(min(rmsError))
alph = alphaList[indexBest]
model = linear_model.Ridge(alpha=alph)
model.fit(xTrain, yTrain)
errorVector = yTest-model.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

plt.scatter(model.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel('Predicted Taste Score')
plt.ylabel('Actual Taste Score')
plt.show()
