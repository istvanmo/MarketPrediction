from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
from time import sleep

N = 2000

x_train = np.reshape(np.linspace(0, 10, N), (-1, 1))
y_train = np.reshape(np.sin(x_train), (-1, 1))
sess = tf.Session()
ajka = np.ones((1, 5), dtype=np.float32)
with sess.as_default():
    init = tf.constant_initializer(ajka)
    net = tflearn.input_data(shape=[None, 1])
    net = tflearn.fully_connected(net, 5, activation='sigmoid', weights_init=init)
    net = tflearn.fully_connected(net, 1, activation='sigmoid')
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001, name="output1")

    model = tflearn.DNN(net, tensorboard_verbose=2)
    # model.fit(x_train, y_train, n_epoch=1, validation_set=0.1, batch_size=50, show_metric=True)
    # ----------------------------------------------------------------------------------------------------------

    trainable_vars_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_vars = model.session.run(trainable_vars_op)
    shape_list = []
    shape_len_list = []

    for i in trainable_vars:
        i_shape_array = np.array(i.shape)
        shape_len = np.prod(i_shape_array)
        shape_len_list.append(shape_len)
        shape_list.append(i.shape)
    print(shape_list)

    # itt csinálni egy megfelelő shapepel rendelkező vektrot

#    ran_vector = np.ones(sum(shape_len_list), dtype=np.float64)
#    print(shape_list)
#    print(shape_len_list)
#    print(sum(shape_len_list))
#
#    # assign vector előállítás
#    assign_vector = []
#    t_v_copy = ran_vector.copy()
#    for length, shape in zip(shape_len_list, shape_list):
#        t_v_chunk = t_v_copy[:length]
#        t_v_chunk.astype(np.float64) # nem biztos hogy kell
#        t_v_copy = t_v_copy[length:] # 0 hosszúságú vektort is visszaadja de nem baj elvileg
#        assign_vector.append(np.reshape(t_v_chunk, shape))
#
#        print(":::::::::::")
#        print(t_v_copy)
#        print(",,,,,,,,,")
#        print(assign_vector)
#
#    v_l = tflearn.variables.get_all_trainable_variable()
#    print(v_l)
#    for act_v, value in zip(v_l, assign_vector):
#        tflearn.variables.set_value(act_v, value, session=sess)
#        print("egyszer")

# ----------------------------------------------------------------------------------------------------

    trainable_vars_op_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_vars_b = model.session.run(trainable_vars_op_b)
    for i in trainable_vars_b:
        print(i)
    sleep(2)

    model.fit(x_train, y_train, n_epoch=1, validation_set=0.1, batch_size=50, show_metric=True)

    trainable_vars_op_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_vars_b = model.session.run(trainable_vars_op_b)
    for i in trainable_vars_b:
        print(i)