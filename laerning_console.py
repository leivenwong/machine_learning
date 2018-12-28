import urllib.request
from sklearn import datasets, preprocessing, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl
from math import sqrt
import pymysql
import numpy
import pandas as pd
import matplotlib.pyplot as plot

from learning_settings import Settings
import learning_functions as fuc


from sklearn.externals import joblib

Modle = joblib.load('GBM.pkl')
test = [[1,2,3,4,5,6,7,8,9,10]]
predictions = Modle.predict(test)
print(predictions)