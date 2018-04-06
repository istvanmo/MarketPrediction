import numpy as np
import tflearn
import tensorflow as tf
from time import sleep
from GetTrainData import train_valid_data


class LossFitCallback(tflearn.callbacks.Callback):
    def __init__(self):
        self.fit_val = None

    def on_train_end(self, training_state):
        loss = training_state.global_loss
        self.fit_val = 1 / loss

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

        # bulid the tf graph
        self.net = tflearn.input_data(shape=[None, 1, 9])
        self.net = tflearn.fully_connected(self.net, self.f_layer_num, activation='tanh')
        self.net = tflearn.fully_connected(self.net, 1, activation='sigmoid')
        self.opt = tflearn.optimizers.Adam(learning_rate=self.l_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                      name='Adam')
        # self.opt = tflearn.optimizers.Nesterov(learning_rate=self.l_rate, momentum=self.momentum, lr_decay=0.95, decay_step=100,
        #                                       staircase=False, use_locking=False, name='Nesterov')
        self.net = tflearn.regression(self.net, optimizer=self.opt, loss='mean_square', name="output")
        self.model = tflearn.DNN(self.net, tensorboard_verbose=0)

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

        self.LFcb = LossFitCallback()

    def create_assign_vector(self, p):
        assign_vector = []
        p_copy = p.copy()
        for length, shape in zip(self.shape_len_list, self.shape_list):
            p_chunk = p_copy[:length]
            p_chunk.astype(np.float64)  # nem biztos hogy kell
            p_copy = p_copy[length:]  # 0 hosszúságú vektort is visszaadja de nem baj elvileg
            assign_vector.append(np.reshape(p_chunk, shape))
        return assign_vector

    def assign_params(self, dna):
        v_l = tflearn.variables.get_all_trainable_variable()
        for act_v, value in zip(v_l, dna):
            tflearn.variables.set_value(act_v, value, session=self.model.session)

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
        return same_counter / len(pred_move)

    def train_all(self, pop):
        fitness_values = []
        dna_num = 0
        for p in pop:
            dna_num += 1
            print("A populácio ennyiedik tagja: ", dna_num)

            assign_vector = self.create_assign_vector(p)

            self.assign_params(assign_vector)

            # trainable_vars_op_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # trainable_vars_b = self.model.session.run(trainable_vars_op_b)
            # for i in trainable_vars_b:
            #     print(i)
            # sleep(2)
            # print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")

            self.model.fit(self.x_train, self.y_train, n_epoch=self.n_epoch, validation_set=(self.x_valid, self.y_valid),
                           batch_size=self.b_size, show_metric=False, snapshot_epoch=False, callbacks=self.LFcb)

            # trainable_vars_op_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # trainable_vars_b = self.model.session.run(trainable_vars_op_b)
            # for i in trainable_vars_b:
            #     print(i)
            # sleep(2)
            # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

            # fit = self.back_t_fit()
            # fitness_values.append(fit)
            fitness_values.append(self.LFcb.fit_val)

        return fitness_values

    def train_one(self, indiv):
        assign_vector = self.create_assign_vector(indiv)

        self.assign_params(assign_vector)

        self.model.fit(self.x_train, self.y_train, n_epoch=self.n_epoch, validation_set=(self.x_valid, self.y_valid),
                       batch_size=self.b_size, show_metric=False, snapshot_epoch=False, callbacks=self.LFcb)
        # fitness_val = self.LFcb.fit_val
        fitness_val = 1 / self.back_t_fit()
        return fitness_val

"""def mean_absolute(y_pred, y_true):
    with tf.name_scope("MeanAbsoluteError"):
        abs_diff = tf.abs(y_pred - y_true)
        mae = tf.reduce_mean(abs_diff)
    return mae"""
