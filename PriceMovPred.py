from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from DataDownload import get_close_prices
from time import sleep


def OBV(c_p_d, v_d):
    obv_list = []
    data_len = len(c_p_d)
    for i in range(data_len):
        if i == 0:
            obv_list.append(v_d[0])
        else:
            theta = 0
            if c_p_d[i] - c_p_d[i-1] > 0:
                theta = 1
            elif c_p_d[i] - c_p_d[i-1] < 0:
                theta = -1

            obv_t = obv_list[i-1] + theta * v_d[i]
            obv_list.append(obv_t)
    return obv_list

def MA5(c_p_d):
    ma5_list = []
    data_len = len(c_p_d)
    for i in range(5, data_len):
        ma5_t = sum(c_p_d[i-5:i]) / 5
        ma5_list.append(ma5_t)
    return ma5_list

def BIASn(max_len):
    pass

def PSYn(n):
    pass

def ASYn(n):
    pass

def train_data(c_p_d):
    pass

time_cprice_data = get_close_prices("USDT_ETH", "1800")
transposed_data = time_cprice_data.T
cprice_data = transposed_data[1]

