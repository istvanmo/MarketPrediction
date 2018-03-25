from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn

N = 2000

x_train = np.reshape(np.linspace(0, 10, N), (-1, 1))
y_train = np.reshape(np.sin(x_train), (-1, 1))

with tf.variable_scope("alap"):
    net = tflearn.input_data(shape=[None, 1])
    net = tflearn.fully_connected(net, 5, activation='sigmoid')
    net = tflearn.fully_connected(net, 1, activation='sigmoid')
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001, name="output1")

model = tflearn.DNN(net, tensorboard_verbose=2)
model.fit(x_train, y_train, n_epoch=100, validation_set=0.1, batch_size=50, show_metric=True)

a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "alap")
b = model.session.run(a)
print(b)
for i in b:
    print(i.shape)
