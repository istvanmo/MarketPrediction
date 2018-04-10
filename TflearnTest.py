from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
from time import sleep

N = 2000

x_train = np.reshape(np.linspace(0, 10, N), (-1, 1))
y_train = np.reshape(np.sin(x_train), (-1, 1))

net = tflearn.input_data(shape=[None, 1])
net = tflearn.fully_connected(net, 5, activation='sigmoid')
net = tflearn.fully_connected(net, 1, activation='sigmoid')
net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001, name="output1")

model = tflearn.DNN(net, tensorboard_verbose=2)
# model.fit(x_train, y_train, n_epoch=1, validation_set=0.1, batch_size=50, show_metric=True)
# ----------------------------------------------------------------------------------------------------------

trainable_vars_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
trainable_vars = model.session.run(trainable_vars_op)

print("Ezek az eredeti súlyok:")
for i in trainable_vars:
    print(i)
print("--------------------------------------------------------------------------------")

assign_val_ph = tf.placeholder(shape=None, dtype=tf.float32)
assign_list = []
for i in trainable_vars_op:
    assign_list.append(tf.assign(i, assign_val_ph))

shape_list = []
shape_len_list = []

for i in trainable_vars:
    i_shape_array = np.array(i.shape)
    shape_len = np.prod(i_shape_array)
    shape_len_list.append(shape_len)
    shape_list.append(i.shape)

print("Ez a súlyok shapeja:")
print(shape_list)
print("--------------------------------------------------------------------------------")

    # itt csinálni egy megfelelő shapepel rendelkező vektrot

ran_vector = np.ones(sum(shape_len_list), dtype=np.float64)
# print(shape_list)
# print(shape_len_list)
# print(sum(shape_len_list))
# assign vector előállítás
assign_vector = []
t_v_copy = ran_vector.copy()
for length, shape in zip(shape_len_list, shape_list):
    t_v_chunk = t_v_copy[:length]
    t_v_chunk.astype(np.float64) # nem biztos hogy kell
    t_v_copy = t_v_copy[length:] # 0 hosszúságú vektort is visszaadja de nem baj elvileg
    assign_vector.append(np.reshape(t_v_chunk, shape))
    # print(":::::::::::")
    # print(t_v_copy)
    # print(",,,,,,,,,")
    # print(assign_vector)

# ----------------------------------------------------------------------------------------------------
for assi, val in zip(assign_list, assign_vector):
    model.session.run(assi, feed_dict={assign_val_ph: val})


trainable_vars_op_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
trainable_vars_b = model.session.run(trainable_vars_op_b)
print("Ezek az assignolt súlyok:")
for i in trainable_vars_b:
    print(i)
print("--------------------------------------------------------------------------------")

model.fit(x_train, y_train, n_epoch=1, validation_set=0.1, batch_size=50, show_metric=True)


trainable_vars_op_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
trainable_vars_b = model.session.run(trainable_vars_op_b)
print("Ezek a tanított súlyok:")
for i in trainable_vars_b:
    print(i)
