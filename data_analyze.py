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
target_raw = fuc.read_sql_wang2(ai_settings)
target_raw = pd.DataFrame(target_raw)

#select close price for compute
data_open = target_raw.loc[0:, ai_settings.fetch_open]
data_high = target_raw.loc[0:, ai_settings.fetch_high]
data_low = target_raw.loc[0:, ai_settings.fetch_low]
data_close = target_raw.loc[0:, ai_settings.fetch_close]
data_date = target_raw.loc[0:, ai_settings.fetch_date]
data_vol = target_raw.loc[0:, ai_settings.fetch_vol]
#data_date = pd.to_datetime(data_date)
#data_date = fuc.to_date(data_date)

#add index for analyze
roll = 1
data_close_roll = fuc.compute_roll(data_close,roll)
data_open_roll = fuc.compute_roll(data_open,roll)
data_high_roll = fuc.compute_roll(data_high,roll)
data_low_roll = fuc.compute_roll(data_low,roll)
macd = fuc.compute_macd(data_close, 12, 26, 9)
macd = fuc.compute_roll(macd, roll)
rsi = fuc.compute_rsi(data_close,9)
rsi = fuc.compute_roll(rsi, roll)
ema = fuc.compute_ema(data_close, 26)
ema = fuc.compute_roll(ema, roll)
open_jump = fuc.open_jump(data_close,data_open)
profit_per_incycle = fuc.profit_per_incycle(data_close, data_open)
profit_per = fuc.profit_per(data_close)
profit_per_roll = fuc.compute_roll(profit_per, roll)
close_square = np.array(data_close) ** 2
close_square_roll = fuc.compute_roll(close_square, roll)

target = pd.DataFrame()
target['open_price'] = data_open_roll
target['high_price'] = data_high_roll
target['low_price'] = data_low_roll
target['close_price'] = data_close_roll
target['macd'] = macd
target['rsi'] = rsi
target['ema'] = ema
target['open_jump'] = open_jump
target['profit_per_roll'] = profit_per_roll
target['profit_per_incycle'] = profit_per_incycle
target['vol'] = data_vol
target['close_square_roll'] = close_square_roll
target['profit_per'] = profit_per

print("data len: " + str(len(data_date)))

if __name__ == '__main__':
    corMat = target.corr()
    print(corMat)
    print(target['close_square_roll'])
    plt.pcolor(corMat)
    plt.show()