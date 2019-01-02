from sklearn.model_selection import train_test_split
import numpy
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

def use_trained_model(for_test):
    Modle = joblib.load('D:\\python_project\\machine_learning\\'
                    'trained_models\\' + ai_settings.fetch_table + 'GBM.pkl')
    test = for_test
    predictions = Modle.predict(test)
    return predictions