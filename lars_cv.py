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

#Normalize columns in x and labels

nrows = len(xList)
ncols = len(xList[0])

#calculate means and variances
xMeans = []
xSD = []
for i in range(ncols):
    col = [xList[j][i] for j in range(nrows)]
    mean = sum(col)/nrows
    xMeans.append(mean)
    colDiff = [(xList[j][i] - mean) for j in range(nrows)]
    sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrows)])
    stdDev = sqrt(sumSq/nrows)
    xSD.append(stdDev)

#use calculated mean and standard deviation to normalize xList
xNormalized = []
for i in range(nrows):
    rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
    xNormalized.append(rowNormalized)

#Normalize labels
meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)

labelNormalized = [(labels[i] - meanLabel)/sdLabel for i in range(nrows)]

#Build cross-validation loop to determine best coefficient values.

#number of cross validation folds
nxval = 10

#number of steps and step size
nSteps = 350
stepSize = 0.004

#initialize list for storing errors.
errors = []
for i in range(nSteps):
    b = []
    errors.append(b)


for ixval in range(2, nxval):
    #Define test and training index sets
    cut = int(len(labels) / ixval)

    #Define test and training attribute and label sets
    xTest = xNormalized[cut:]
    xTrain = xNormalized[:cut]
    labelTest = labelNormalized[cut:]
    labelTrain = labelNormalized[:cut]

    #Train LARS regression on Training Data
    nrowsTrain = len(labelTrain)
    nrowsTest = len(labelTest)

    #initialize a vector of coefficients beta
    beta = [0.0] * ncols

    #initialize matrix of betas at each step
    betaMat = []
    betaMat.append(list(beta))

    for iStep in range(nSteps):
        #calculate residuals
        residuals = [0.0] * nrows
        for j in range(nrowsTrain):
            labelsHat = sum([xTrain[j][k] * beta[k] for k in range(ncols)])
            residuals[j] = labelTrain[j] - labelsHat

        #calculate correlation between attribute columns from normalized wine and residual
        corr = [0.0] * ncols

        for j in range(ncols):
            corr[j] = sum([xTrain[k][j] * residuals[k] for k in range(nrowsTrain)]) / nrowsTrain

        iStar = 0
        corrStar = corr[0]

        for j in range(1, (ncols)):
            if abs(corrStar) < abs(corr[j]):
                iStar = j; corrStar = corr[j]

        beta[iStar] += stepSize * corrStar / abs(corrStar)
        betaMat.append(list(beta))

        #Use beta just calculated to predict and accumulate out of sample error - not being used in the calc of beta
        for j in range(nrowsTest):
            labelsHat = sum([xTest[j][k] * beta[k] for k in range(ncols)])
            err = labelTest[j] - labelsHat
            errors[iStep].append(err)

cvCurve = []
for errVect in errors:
    mse = sum([x*x for x in errVect])/len(errVect)
    cvCurve.append(mse)

minMse = min(cvCurve)
minPt = [i for i in range(len(cvCurve)) if cvCurve[i] == minMse ][0]
print("Minimum Mean Square Error", minMse)
print("Index of Minimum Mean Square Error", minPt)
print("Best beta", betaMat[minPt])

xaxis = range(len(cvCurve))
plt.plot(xaxis, cvCurve)

plt.xlabel("Steps Taken")
plt.ylabel(("Mean Square Error"))
plt.show()
