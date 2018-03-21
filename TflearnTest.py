from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
import tflearn

N = 2000
seq_len = 50

x_train_lp = np.linspace(0, 10, N)
x_train = view_as_windows(x_train_lp, (seq_len,))
x_train = np.reshape(x_train[:-1], (-1, 1, seq_len))
y_train = np.sin(x_train_lp) * x_train_lp / 2
y_train = np.reshape(y_train[seq_len:], (-1, 1))

net = tflearn.input_data(shape=[None, 1, seq_len])
net = tflearn.lstm(net, 128, return_seq=False)
net = tflearn.fully_connected(net, 1, activation='linear')
net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001, name="output1")

model = tflearn.DNN(net, tensorboard_verbose=2)
model.fit(x_train, y_train, n_epoch=10000, validation_set=0.1, batch_size=50, show_metric=True)