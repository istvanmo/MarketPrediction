"""def mean_absolute(y_pred, y_true):
    with tf.name_scope("MeanAbsoluteError"):
        abs_diff = tf.abs(y_pred - y_true)
        mae = tf.reduce_mean(abs_diff)
    return mae
    
    def train_all(self, pop):
    fitness_values = []
    dna_num = 0
    for p in pop:
        dna_num += 1
        print("A populácio ennyiedik tagja: ", dna_num)
        assign_vector = self.create_assign_vector(p)
        self.assign_params(assign_vector)
        self.model.fit(self.x_train, self.y_train, n_epoch=self.n_epoch, validation_set=(self.x_valid, self.y_valid),
                       batch_size=self.b_size, show_metric=False, snapshot_epoch=False, callbacks=self.LFcb)
        # fit = self.back_t_fit()
        # fitness_values.append(fit)
        fitness_values.append(self.LFcb.fit_val)
        return fitness_values
    
    """

import numpy as np
from time import sleep
import keras
from GetTrainData import train_valid_data
from keras.models import Sequential
from keras.layers import Dense, Dropout

class BP_ANN:
    def __init__(self, f_layer_num, l_rate, momentum, n_epoch, batch_size, valid_rate, do, opt="adam"):
        # shape of the init weights
        # (9,10); (10,); (10,1); (10,)
        self.f_layer_num = f_layer_num
        self.l_rate = l_rate
        self.momentum = momentum
        self.n_epoch = n_epoch
        self.b_size = batch_size
        self.x_train, self.y_train_h, self.x_test, self.y_test_h, self.cp = train_valid_data(valid_rate)
        # y needs to be reshaped
        self.y_train = np.reshape(self.y_train_h, (-1, 1, 2))
        self.y_test = np.reshape(self.y_test_h, (-1, 1, 2))

        # bulid the tf graph
        self.model = Sequential()

        h_layer = Dense(4, activation='relu', input_shape=(1, 9), name="Hidden_layer")
        o_layer = Dense(2, activation='softmax', name="Output_layer")

        self.model.add(h_layer)
        self.model.add(o_layer)

        self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

        self.layer_list = [h_layer, o_layer]

        self.l_w_shape_list, self.l_w_len_list = self.variable_shapes(self.layer_list)

    def variable_shapes(self, layer_l):
        l_w_shape_list = []
        l_w_len_list = []
        for layer in layer_l:
            w_of_layer = layer.get_weights()
            l_w_shape = []
            l_w_len = []
            for par in w_of_layer:
                shape = par.shape
                len = np.prod(np.ravel(shape))
                l_w_shape.append(shape)
                l_w_len.append(len)
            l_w_shape_list.append(l_w_shape)
            l_w_len_list.append(l_w_len)
        return l_w_shape_list, l_w_len_list

    def create_assign_vector(self, p, l_w_shape_list, l_w_len_list):
        assign_vec = []
        for s, l in zip(l_w_shape_list, l_w_len_list):
            h_assign_vector = []
            p_copy = p[:np.sum(l)].copy()
            for length, shape in zip(l, s):
                p_chunk = p_copy[:length]
                p_chunk.astype(np.float32)  # nem biztos hogy kell
                p_copy = p_copy[length:]  # 0 hosszúságú vektort is visszaadja
                if len(p_chunk) == 0:
                    break
                h_assign_vector.append(np.reshape(p_chunk, shape))
            assign_vec.append(h_assign_vector)
        return assign_vec

    def assign_variables(self, layer_list, a_vect):
        for lay, assign in zip(layer_list, a_vect):
            lay.set_weights(assign)

    def back_t_fit(self, indiv):
        assign_vector = self.create_assign_vector(indiv, self.l_w_shape_list, self.l_w_len_list)
        self.assign_variables(self.layer_list, assign_vector)
        pred_move = self.model.predict(self.x_test)
        hit_counter = 0
        for i in range(len(pred_move)):
            choice = np.random.choice((0, 1), 1, p=pred_move[i][0])
            if choice[0] == self.y_test[i][0][1]:
                hit_counter += 1

        return len(self.x_test) - hit_counter # gyakorlatilag a hibák száma -> ezt kell 0-ra vinni

    def train_one(self, indiv, is_bp):
        fitness_val = None

        # Only backpropagation
        if indiv is None and is_bp is True:
            print("---- ONLY BACKPROPAGATION ----")
            sleep(2)
            tbCallBack = keras.callbacks.TensorBoard(log_dir='./log_dir', histogram_freq=0, write_graph=True,
                                                     write_images=True)
            # itt nem kell indiv a keras saját inicializálása tanul
            self.model.fit(self.x_train, self.y_train, batch_size=self.b_size, epochs=self.n_epoch,
                           callbacks=[tbCallBack])

            fitness_val = self.model.evaluate(self.x_test, self.y_test, batch_size=128)

        # Full
        if indiv is not None and is_bp is True:
            assign_vector = self.create_assign_vector(indiv, self.l_w_shape_list, self.l_w_len_list)
            self.assign_variables(self.layer_list, assign_vector)
            self.model.fit(self.x_train, self.y_train, batch_size=self.b_size, epochs=self.n_epoch)

            fitness_val = self.model.evaluate(self.x_test, self.y_test, batch_size=128)

        # Only Differential Evolution or Genetic algorithm
        if indiv is not None and is_bp is False:
            # assign is in the backtest
            fitness_val = self.back_t_fit(indiv)

        return fitness_val
