import json
import os
import numpy as np
# import pandas as pd
# import h5py
from urllib.request import urlopen

def get_data(coin_pair, period):
    if os.path.exists("data.json"):
        with open("data.json") as data_file:
            data = json.load(data_file)
    else:
        with open('data.json', 'w+') as outfile:
            url = "https://poloniex.com/public?command=returnChartData&currencyPair=" +coin_pair+ "&start=1490054400&end=9999999999&period=" +period
            r = urlopen(url)
            data = json.loads(r.read().decode(encoding='UTF-8'))
            json.dump(data, outfile)
    return data


def get_close_prices(pair, period):
    full_data = get_data(pair, period)
    time_cprice_list = []
    last = 0
    for i in full_data:
        time = i["date"]
        # not the nicest one
        if time < last:
            print("ORDER FAILURE")
            break
        # end of the not the nicest one
        close = i["close"]
        time_cprice_list.append([time, close])
        last = time
    return np.asarray(time_cprice_list)
