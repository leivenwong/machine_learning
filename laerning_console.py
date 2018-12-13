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

print(data[2][3])