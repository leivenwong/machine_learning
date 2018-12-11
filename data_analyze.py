import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from learning_settings import Settings
import learning_functions as fuc

import sys
sys.path.append('D:\\python_project\\statistics')
from statistics_functions import compute_r

#initiate settings
ai_settings = Settings()

#read raw data
target = fuc.read_sql_merged(ai_settings)
target = pd.DataFrame(target)

#select close price for compute
data_open = target.loc[0:, ai_settings.fetch_open]
data_high = target.loc[0:, ai_settings.fetch_high]
data_low = target.loc[0:, ai_settings.fetch_low]
data_close = target.loc[0:, ai_settings.fetch_close]
data_date = target.loc[0:, ai_settings.fetch_date]
#data_date = pd.to_datetime(data_date)
#data_date = fuc.to_date(data_date)

macd = fuc.compute_macd(data_close, 12, 26, 9)
rsi = fuc.compute_rsi(data_close,9)
ema = fuc.compute_ema(data_close, 26)
open_jump = fuc.open_jump(data_close,data_open)
profit_per_incycle = fuc.profit_per_incycle(data_close, data_open)
profit_per = fuc.profit_per(data_close)

target['macd'] = macd
target['rsi'] = rsi
target['ema'] = ema
target['open_jump'] = open_jump
target['profit_per_incycle'] = profit_per_incycle
target['profit_per'] = profit_per

if __name__ == '_main__':
    corMat = target.corr()
    print(corMat)
    plt.pcolor(corMat)
    plt.show()