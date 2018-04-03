import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from time import sleep
from GetTrainData import train_valid_data


class BP_ANN:
    def __init__(self, f_layer_num, l_rate, momentum, n_epoch, batch_size, valid_rate):
        # shape of the init weights
        # (9,10); (10,); (10,1); (10,)
        self.f_layer_num = f_layer_num
        self.l_rate = l_rate
        self.momentum = momentum
        self.n_epoch = n_epoch
        self.b_size = batch_size
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.cp = train_valid_data(valid_rate)
        print(len(self.x_valid))
        print(len(self.y_valid))

        # bulid the tf graph
        self.net = tflearn.input_data(shape=[None, 1, 9])
        self.net = tflearn.fully_connected(self.net, self.f_layer_num, activation='relu', regularizer="L2")
        # net = tflearn.dropout(net, 0.7)
        self.net = tflearn.fully_connected(self.net, 1, activation='sigmoid')
        self.opt = tflearn.optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                      name='Adam')
        # opt = tflearn.optimizers.Momentum(learning_rate=0.1, momentum=0.4, lr_decay=0.0, decay_step=100,staircase=False, use_locking=False, name='Momentum')
        self.net = tflearn.regression(self.net, optimizer=self.opt, loss='mean_square', name="output1")
        self.model = tflearn.DNN(self.net, tensorboard_verbose=2)

        # extract the shapes of the variables

        trainable_vars_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_vars = self.model.session.run(trainable_vars_op)
        self.shape_list = []
        self.shape_len_list = []

        for i in trainable_vars:
            i_shape_array = np.array(i.shape)
            shape_len = np.prod(i_shape_array)
            self.shape_len_list.append(shape_len)
            self.shape_list.append(i.shape)

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
        return same_counter

    def train_all(self, pop):
        fitness_values = []
        for p in pop:
            assign_vector = []
            p_copy = p.copy()
            for length, shape in zip(self.shape_len_list, self.shape_list):
                p_chunk = p[:length]
                p_chunk.astype(np.float64) # nem biztos hogy kell
                p_copy = p_copy[length:] # 0 hosszúságú vektort is visszaadja de nem baj elvileg
                assign_vector.append(np.reshape(p_chunk, shape))

            self.create_graph(assign_vector[0], assign_vector[1], assign_vector[2], assign_vector[3])
            self.model.fit(self.x_train, self.y_train, n_epoch=self.n_epoch, validation_set=(self.x_valid, self.y_valid),
                           batch_size=self.b_size, show_metric=False)
            fit = self.back_t_fit()
            fitness_values.append(fit)

        return fitness_values

    def create_graph(self, f_l_w, f_l_b, s_l_w, s_l_b):
        init_f_l_w = tf.constant_initializer(f_l_w)
        init_f_l_b = tf.constant_initializer(f_l_b)
        init_s_l_w = tf.constant_initializer(s_l_w)
        init_s_l_b = tf.constant_initializer(s_l_b)
        # bulid the tf graph
        self.net = tflearn.input_data(shape=[None, 1, 9])
        self.net = tflearn.fully_connected(self.net, self.f_layer_num, activation='relu', weights_init=init_f_l_w,
                                           bias_init=init_f_l_b)
        # net = tflearn.dropout(net, 0.7)
        self.net = tflearn.fully_connected(self.net, 1, activation='sigmoid', weights_init=init_s_l_w,
                                           bias_init=init_s_l_b)
        self.opt = tflearn.optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                           use_locking=False,
                                           name='Adam')
        # opt = tflearn.optimizers.Momentum(learning_rate=0.1, momentum=0.4, lr_decay=0.0, decay_step=100,staircase=False, use_locking=False, name='Momentum')
        self.net = tflearn.regression(self.net, optimizer=self.opt, loss='mean_square', name="output1")
        self.model = tflearn.DNN(self.net, tensorboard_verbose=2)

"""def mean_absolute(y_pred, y_true):
    with tf.name_scope("MeanAbsoluteError"):
        abs_diff = tf.abs(y_pred - y_true)
        mae = tf.reduce_mean(abs_diff)
    return mae"""
