class Settings():
    """initiate settings"""
    def __init__(self):
        #read excel file' path
        self.file_path = "macro.xlsx"

        #read Mysql database's path
        self.sql_path_merged = 'mysql+pymysql://ctp_user:ctp_password' \
            '@127.0.0.1/ctp_merged_mq?charset=utf8'

        self.sql_path_backtesting = 'mysql+pymysql://ctp_user:ctp_password' \
            '@127.0.0.1/ctp_backtesting?charset=utf8'

        self.sql_path_wang2 = 'mysql+pymysql://wang_2:wang_2' \
            '@127.0.0.1/python_merge?charset=utf8'

        #set which table will used in mysql database
        self.fetch_table = 'ru_1d'

        # fetch close price in raw data
        self.fetch_close = "close_price"

        #fetch open
        self.fetch_open = "open_price"

        #fetch high
        self.fetch_high = "high_price"

        #fetch low
        self.fetch_low = "low_price"

        # fetch date column in raw data
        self.fetch_date = "utc_string"

        #fetch volumn
        self.fetch_vol = "volumn"

        #set trade fee
        self.trade_fee = 0.0000325

        #set leverage
        self.leverage_rate = 1

        #if stop
        self.stop = 0.2

        #if stopwin
        self.stopwin = 0.6

        #if jump close and open
        self.jump_night = False

        #if only buy
        self.only_buy = True

        #if want to draw polt
        self.draw_plot = True

