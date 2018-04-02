import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from time import sleep
from GetTrainData import train_valid_data


class BP_ANN:
    def __init__(self, l_rate, momentum, n_epoch, batch_size, valid_rate):
        self.l_rate = l_rate
        self.momentum = momentum
        self.n_epoch = n_epoch
        self.b_size = batch_size
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.cp = train_valid_data(valid_rate)

        # bulid the tf graph
        self.net = tflearn.input_data(shape=[None, 1, 9])
        self.net = tflearn.fully_connected(self.net, 10, activation='relu', regularizer="L2")
        # net = tflearn.dropout(net, 0.7)
        self.net = tflearn.fully_connected(self.net, 1, activation='sigmoid')
        self.opt = tflearn.optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                      name='Adam')
        # opt = tflearn.optimizers.Momentum(learning_rate=0.1, momentum=0.4, lr_decay=0.0, decay_step=100,staircase=False, use_locking=False, name='Momentum')
        self.net = tflearn.regression(self.net, optimizer=self.opt, loss='mean_square', name="output1")
        self.model = tflearn.DNN(self.net, tensorboard_verbose=2)

        # TODO: extract the shapes of the variables

        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        b = model.session.run(a)
        shape_list = []
        shape_len_list = []
        for i in b:
            i_shape_array = np.array(i.shape)
            shape_len = np.prod(i_shape_array)
            shape_len_list.append(shape_len)
            shape_list.append(i.shape)

    def back_t_fit(self):
        pred_move = self.model.predict(self.x_valid)
        f_c_p = np.array(self.cp[1:])
        n_c_p = np.array(self.cp[:-1])
        real_move_all = f_c_p - n_c_p
        real_move = real_move_all[-len(pred_move):]
        y_backtest = []
        for m in real_move:
            if m > 0:
                y_backtest.append(1)
            else:
                y_backtest.append(0)

        same_counter = 0
        for re, ann in zip(y_backtest, pred_move):
            dis = np.abs(re - ann)
            if dis < 0.5:
                same_counter += 1
        pass

    def train_all(self, pop):
        fitness_values = []
        # TODO: first train one by one on only one thread
        # iterate over the pop, reshape the weights, assing them to the variables of the graph
        self.model.fit(self.x_train, self.y_train, n_epoch=self.n_epoch, validation_set=(self.x_valid, self.y_valid),
                       batch_size=self.b_size, show_metric=True)
        return fitness_values


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