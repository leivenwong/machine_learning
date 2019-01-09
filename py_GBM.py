import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import numpy
import matplotlib.pyplot as plot
from learning_settings import Settings
import data_analyze
from sklearn.externals import joblib


#initiate settings
ai_settings = Settings()

#read raw data
target_raw = data_analyze.target

#here are some made-up numbers to start with
data = target_raw[['open_price', 'high_price', 'low_price', 'close_price',
                   'macd', 'rsi', 'ema', 'profit_per_roll', 'vol',
                   'close_square_roll']].values.tolist()
prediction = list(target_raw['profit_per'])

xList = data
labels = prediction
names = ['open_price', 'high_price', 'low_price', 'close_price', 'macd',
         'rsi', 'ema', 'profit_per_roll', 'vol',
         'close_square_roll', 'profit_per']

nrows = len(xList)
ncols = len(xList[0])

X = numpy.array(xList)
y = numpy.array(labels)
wineNames = numpy.array(names)

#take fixed holdout set 30% of data rows
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30,
                                                random_state=531)
train_len = len(xTrain)
print('train_len:' + str(train_len))
backtesting_data = pd.DataFrame()
backtesting_data['open_price'] = xTest[0:][0]
backtesting_data['high_price'] = xTest[0:][1]
backtesting_data['low_price'] = xTest[0:][2]
backtesting_data['close_price'] = xTest[0:][3]

# Train gradient boosting model to minimize mean squared error
nEst = 60
depth = 5
learnRate = 0.01
subSamp = 0.5
GBMModel = ensemble.GradientBoostingRegressor(n_estimators=nEst,
                                                  max_depth=depth,
                                                  learning_rate=learnRate,
                                                  subsample = subSamp,
                                                  loss='ls')

GBMModel.fit(xTrain, yTrain)

# compute mse on test set
msError = []
predictions = GBMModel.staged_predict(xTest)
for p in predictions:
    msError.append(mean_squared_error(yTest, p))

joblib.dump(GBMModel, 'D:\\python_project\\machine_learning\\'
                      'trained_models\\' + ai_settings.fetch_table +
                        'GBM.pkl')
print("MSE")
print(min(msError))
print(msError.index(min(msError)))

d_prediction = GBMModel.predict(xTest)

print("prediction length:" +
      str(len(d_prediction)))

if __name__ == '__main__':
    #plot training and test errors vs number of trees in ensemble
    plot.figure()
    plot.plot(range(1, nEst + 1), GBMModel.train_score_, label='Training Set MSE')
    plot.plot(range(1, nEst + 1), msError, label='Test Set MSE')
    plot.legend(loc='upper right')
    plot.xlabel('Number of Trees in Ensemble')
    plot.ylabel('Mean Squared Error')
    plot.show()

    # Plot feature importance
    featureImportance = GBMModel.feature_importances_

    # normalize by max importance
    featureImportance = featureImportance / featureImportance.max()
    idxSorted = numpy.argsort(featureImportance)
    barPos = numpy.arange(idxSorted.shape[0]) + .5
    plot.barh(barPos, featureImportance[idxSorted], align='center')
    plot.yticks(barPos, wineNames[idxSorted])
    plot.xlabel('Variable Importance')
    plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    plot.show()