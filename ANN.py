import numpy as np
from time import sleep, time
import keras
from GetTrainData import train_valid_data
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam

class BP_ANN:
    def __init__(self, h_layer_num, l_rate, momentum, n_epoch, batch_size, test_rate, do, opt="adam"):
        # shape of the init weights
        # (9,10); (10,); (10,1); (10,)
        self.h_layer_num = h_layer_num
        self.l_rate = l_rate
        self.momentum = momentum
        self.n_epoch = n_epoch
        self.b_size = batch_size
        self.x_train, self.y_train_h, self.x_test, self.y_test_h, self.cp = train_valid_data(test_rate)
        # y needs to be reshaped
        self.y_train = np.reshape(self.y_train_h, (-1, 1, 2))
        self.y_test = np.reshape(self.y_test_h, (-1, 1, 2))

        # BUILD THE GRAPH
        self.model = Sequential()

        # input + hidden layer
        h_layer = Dense(self.h_layer_num, input_shape=(1, 9), kernel_initializer="glorot_normal",
                        name="Hidden_layer")
        h_l_batch_norm = BatchNormalization()
        h_l_activation = Activation("relu")
        if do:
            do_layer = Dropout(0.2)

        # output layer
        o_layer = Dense(2, kernel_initializer="glorot_normal", name="Output_layer")
        o_l_batch_norm = BatchNormalization()
        o_l_activation = Activation("softmax")

        # add layers to model
        self.model.add(h_layer)
        self.model.add(h_l_batch_norm)
        self.model.add(h_l_activation)
        if do:
            self.model.add(do_layer)
        self.model.add(o_layer)
        self.model.add(o_l_batch_norm)
        self.model.add(o_l_activation)

        # optimizer
        optimizer = Adam(lr=self.l_rate)
        if opt == "momentum":
            optimizer = SGD(lr=self.l_rate, momentum=self.momentum)


        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        # create the vector of the shapes of variables -> for the assign
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
                length = np.prod(np.ravel(shape))
                l_w_shape.append(shape)
                l_w_len.append(length)
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
        if indiv is not None:
            assign_vector = self.create_assign_vector(indiv, self.l_w_shape_list, self.l_w_len_list)
            self.assign_variables(self.layer_list, assign_vector)
        pred_move = self.model.predict(self.x_test)
        hit_counter = 0
        for i in range(len(pred_move)):
            choice = np.random.choice((0, 1), 1, p=pred_move[i][0])
            if choice[0] == self.y_test[i][0][1]:
                hit_counter += 1

        return [len(self.x_test) - hit_counter] # gyakorlatilag a hibák száma -> ezt kell 0-ra vinni

    def train_one(self, indiv, is_bp):
        fitness_val = None

        # Only backpropagation
        if indiv is None and is_bp is True:
            print("---- ONLY BACKPROPAGATION ----")
            sleep(2)
            ti = time()
            ti = str(int(ti))
            l_dir = "./log_dir/ts_" + ti[-6:] + "__lr_" + str(self.l_rate) + "__h_" + str(self.h_layer_num)
            tbCallBack = keras.callbacks.TensorBoard(log_dir=l_dir, histogram_freq=0, write_graph=True,
                                                     write_images=True)
            # itt nem kell indiv a keras saját inicializálása tanul
            self.model.fit(self.x_train, self.y_train, batch_size=self.b_size, epochs=self.n_epoch,
                           callbacks=[tbCallBack])

            print("Number of test points: ", len(self.y_test))
            wrong = self.back_t_fit(None)
            print("Wrong: ", wrong[0])
            print("Hit ratio: ", (len(self.y_test) - wrong[0]) / len(self.y_test))

            fitness_val = self.model.evaluate(self.x_test, self.y_test, batch_size=self.b_size)

        # TODO: Check how the BatchNorm layer works in inference. Mostly in only GA
        # Full
        if indiv is not None and is_bp is True:
            assign_vector = self.create_assign_vector(indiv, self.l_w_shape_list, self.l_w_len_list)
            self.assign_variables(self.layer_list, assign_vector)
            self.model.fit(self.x_train, self.y_train, batch_size=self.b_size, epochs=self.n_epoch)

            fitness_val = self.model.evaluate(self.x_test, self.y_test, batch_size=self.b_size)

        # Only Genetic algorithm
        if indiv is not None and is_bp is False:
            # assign is in the backtest
            fitness_val = self.back_t_fit(indiv)

        return fitness_val

