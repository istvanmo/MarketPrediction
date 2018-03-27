# from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
# import tensorflow as tf
from time import sleep
from random import shuffle
from GetIndexData import get_data

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

def MAn(c_p_d, n):
    man_list = []
    c_p_d = np.array(c_p_d)
    data_len = len(c_p_d)
    for i in range(n, data_len):
        man_t = np.sum(c_p_d[i-n:i]) / n
        man_list.append(man_t)
    return man_list

def BIAS6(c_p_d):
    ma6 = np.array(MAn(c_p_d, 6))
    len_ma6 = len(ma6)
    sh_c_p_d = np.array(c_p_d[-len_ma6:])
    bias6_list = ((sh_c_p_d - ma6) / ma6) * 100
    return bias6_list

def PSY12(c_p_d):
    psy12_list = []
    f_c_p_d = np.array(c_p_d[1:])
    n_c_p_d = np.array(c_p_d[:-1])
    move = f_c_p_d - n_c_p_d
    for i in range(11, len(move)):
        window = move[i-11: i]
        psy12 = len([x for x in window if x > 0])
        psy12_list.append(psy12)
    return psy12_list

def ASYn(c_p_d, n):
    asy_list = []
    f_c_p_d = np.array(c_p_d[1:])
    n_c_p_d = np.array(c_p_d[:-1])
    ln_f_c = np.log(f_c_p_d)
    ln_n_c = np.log(n_c_p_d)
    ret = ln_f_c - ln_n_c
    sy = ret * 100
    for i in range(n, len(ln_f_c)):
        window = sy[i-n: i]
        asy = np.mean(window)
        asy_list.append(asy)
    return asy_list

def normal_data(d):
    min_value = np.min(d)
    max_value = np.max(d)
    n_vec = []
    for val in d:
        norm_val = (val - min_value) / (max_value - min_value)
        n_vec.append(norm_val)
    return  np.array(n_vec)

def train_test_data(cp, v, valid_rat):
    f_c_p_d = np.array(cp[1:])
    n_c_p_d = np.array(cp[:-1])
    move = f_c_p_d - n_c_p_d
    y_data = []
    for m in move:
        if m > 0:
            y_data.append(1)
        else:
            y_data.append(0)
    y_data = np.array(y_data)
    y_data = y_data.reshape((-1, 1))

    obv = OBV(cp, v)
    ma5 = MAn(cp, 5)
    bias6 = BIAS6(cp)
    psy12 = PSY12(cp)
    asy5 = ASYn(cp, 5)
    asy4 = ASYn(cp, 4)
    asy3 = ASYn(cp, 3)
    asy2 = ASYn(cp, 2)
    asy1 = ASYn(cp, 1)
    data_vect = [obv, ma5, bias6, psy12, asy5, asy4, asy3, asy2, asy1]
    min_len = np.min(np.array([len(x) for x in data_vect]))
    same_len_datas = []
    for d in data_vect:
        same_len_datas.append(d[-min_len:])

    normalized_data = []
    for d in same_len_datas:
        norm_data = normal_data(d)
        normalized_data.append(norm_data)
    normalized_data = np.array(normalized_data)
    x_data = normalized_data.T

    # split data
    split_index = int(len(x_data) * (1 - valid_rat))
    x_train = x_data[0: split_index]
    x_train = x_train.reshape((-1, 1, len(x_train[0])))
    x_valid = x_data[split_index:]
    x_valid = x_valid.reshape((-1, 1, len(x_valid[0])))
    y_train = y_data[0: split_index]
    y_valid = y_data[split_index:]

    # return x_train, np.array(y_train, dtype=np.float64), x_valid, np.array(y_valid, dtype=np.float64)
    return x_train, y_train, x_valid, y_valid

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

cp, v, d = get_data()

x_train, y_train, x_valid, y_valid = train_test_data(cp, v, 0.996)
# x_valid, y_valid = x_valid[0:5], y_valid[0:5]
x_train, y_train = randomize(x_train, y_train)

net = tflearn.input_data(shape=[None, 1, 9])
fc_1 = tflearn.fully_connected(net, 10, activation='relu')  # , regularizer="L2")
# net = tflearn.dropout(fc_1, 0.7)
fc_2 = tflearn.fully_connected(fc_1, 1, activation='sigmoid')
opt = tflearn.optimizers.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
# opt = tflearn.optimizers.Momentum(learning_rate=0.1, momentum=0.4, lr_decay=0.0, decay_step=100, staircase=False, use_locking=False, name='Momentum')
net = tflearn.regression(fc_2, optimizer=opt, loss='mean_square', batch_size=64, name="output1")

# run training

model = tflearn.DNN(net, tensorboard_verbose=2)
model.fit(x_train, y_train, n_epoch=200, validation_set=(x_valid, y_valid), batch_size=64, show_metric=True)

pred_move = model.predict(x_valid)
print(pred_move)

f_c_p = np.array(cp[1:])
n_c_p = np.array(cp[:-1])
real_move = f_c_p - n_c_p
y_backtest = []
for m in real_move:
    if m > 0:
        y_backtest.append(1)
    else:
        y_backtest.append(0)

y_backtest = y_backtest[-len(pred_move):]

same_counter = 0
for re, ann in zip(y_backtest, pred_move):
    dis = np.abs(re - ann)
    if dis < 0.5:
        same_counter += 1

print(same_counter/len(pred_move))



"""def mean_absolute(y_pred, y_true):
    with tf.name_scope("MeanAbsoluteError"):
        abs_diff = tf.abs(y_pred - y_true)
        mae = tf.reduce_mean(abs_diff)
    return mae"""
