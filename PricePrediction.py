from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
from DataDownload import get_close_prices
from time import sleep

# create train and validation data

seq_len = 30
training_points_num = 10000
validation_points_num = 1000
validation_fraction = 0.08

time_cprice_data = get_close_prices("USDT_ETH")
transposed_data = time_cprice_data.T
cprice_data = transposed_data[1]

random_low_train = seq_len + 1
random_high_train = int(len(cprice_data) - (len(cprice_data) * validation_fraction))
x_train = []
y_train = []
for i in range(training_points_num):
    time = np.random.randint(random_low_train, random_high_train)
    low_time = time-seq_len
    x_train.append(cprice_data[low_time: time])
    y_train.append(cprice_data[time])

random_low_test = int(len(cprice_data) - (len(cprice_data) * validation_fraction))
random_high_test = len(cprice_data)
x_test = []
y_test = []
for i in range(validation_points_num):
    time = np.random.randint(random_low_test, random_high_test)
    low_time = time - seq_len
    x_test.append(cprice_data[low_time: time])
    y_test.append(cprice_data[time])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

max_value = np.max([np.max(x_train), np.max(y_train), np.max(x_test), np.max(y_test)])
print(max_value)
sleep(5)

x_train = x_train / max_value
y_train = y_train / max_value
x_test = x_test / max_value
y_test = y_test / max_value

x_train = np.reshape(x_train, (-1, 1, seq_len))
y_train = np.reshape(y_train, (-1, 1))
x_test = np.reshape(x_test, (-1, 1, seq_len))
y_test = np.reshape(y_test, (-1, 1))

# build network

net = tflearn.input_data(shape=[None, 1, seq_len])
net = tflearn.lstm(net, 128, return_seq=False)
net = tflearn.dropout(net, 0.7)
net = tflearn.fully_connected(net, 1, activation='linear')
net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001, name="output1")

# run training

model = tflearn.DNN(net, tensorboard_verbose=2)
model.fit(x_train, y_train, n_epoch=1, validation_set=(x_test, y_test), batch_size=50, show_metric=True)