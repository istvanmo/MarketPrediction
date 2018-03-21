from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
import tflearn
from DataDownload import get_close_prices

# create train and validation data
seq_len = 30
training_points_num = 10
validation_points_num = 100
validation_fraction = 0.08

time_cprice_data = get_close_prices("USDT_ETH")
transpose_data = time_cprice_data.T
cprice_data = transpose_data[1]

random_low = seq_len + 1
random_high = int(len(time_cprice_data) - (len(time_cprice_data) * validation_fraction))
x_train = []
y_train = []
for i in range(training_points_num):
    time = np.random.randint(random_low, random_high)
    low_time = time-seq_len
    x_train.append(cprice_data[low_time: time])
    print(len(cprice_data[low_time: time]))
    y_train.append(cprice_data[time])