from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from DataDownload import get_close_prices
from time import sleep


def OBV():
    pass

def MAn(c_p_d, n):
    point_num = len(c_p_d)

    pass

def BIASn(n):
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

