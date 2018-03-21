import json
import os
import pandas as pd
import h5py
from urllib.request import urlopen

def get_data(coin_pair):
    if os.path.exists("data.json"):
        # megnyit
        with open("data.json") as data_file:
            data = json.load(data_file)
    else:
        # letölt elment
        with open('data.json', 'w+') as outfile:
            url = "https://poloniex.com/public?command=returnChartData&currencyPair=" +coin_pair+ "&start=1356998100&end=9999999999&period=300"
            r = openUrl.read() # itt még fos
            openUrl.close()
            data = json.loads(r.decode())
            json.dump(data, outfile)
    return None

get_data("USDT_ETH")