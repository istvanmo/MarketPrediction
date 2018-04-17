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
        self.fit_val = loss

class BP_ANN:
    def __init__(self, f_layer_num, l_rate, momentum, n_epoch, batch_size, valid_rate, do, opt="adam"):
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
        if do:
            self.net = tflearn.fully_connected(self.net, self.f_layer_num, activation='relu', regularizer="L2")
            # self.net = tflearn.dropout(self.net, 0.9, noise_shape=None, name='Dropout')
        else:
            self.net = tflearn.fully_connected(self.net, self.f_layer_num, activation='relu')
        self.net = tflearn.fully_connected(self.net, 2, activation='softmax')

        if opt == "momentum":
            self.opt = tflearn.optimizers.Momentum(learning_rate=0.001, momentum=0.9, lr_decay=0.95, decay_step=100,
                                                   staircase=False, use_locking=False, name='Momentum')
        elif opt == "nesterov":
            self.opt = tflearn.optimizers.Nesterov(learning_rate=self.l_rate, momentum=self.momentum, lr_decay=0.95,
                                                   decay_step=100, staircase=False, use_locking=False, name='Nesterov')
        else:
            self.opt = tflearn.optimizers.Adam(learning_rate=self.l_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                               use_locking=False, name='Adam')

        self.net = tflearn.regression(self.net, optimizer=self.opt, loss='categorical_crossentropy', name="output")
        self.model = tflearn.DNN(self.net, tensorboard_verbose=6)

        # extract the shapes of the variables and create assign op-s

        trainable_vars_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_vars = self.model.session.run(trainable_vars_op)

        self.assign_val_ph = tf.placeholder(shape=None, dtype=tf.float32)
        self.assign_list = []
        for t_v in trainable_vars_op:
            self.assign_list.append(tf.assign(t_v, self.assign_val_ph))

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
            p_chunk.astype(np.float32)  # nem biztos hogy kell
            p_copy = p_copy[length:]  # 0 hosszúságú vektort is visszaadja de nem baj elvileg
            if len(p_chunk) == 0:
                break
            assign_vector.append(np.reshape(p_chunk, shape))
        return assign_vector

    def assign_params(self, dna):
        for assign_op, val in zip(self.assign_list, dna):
            self.model.session.run(assign_op, feed_dict={self.assign_val_ph: val})

    def back_t_fit(self, indiv):
        assign_vector = self.create_assign_vector(indiv)
        self.assign_params(assign_vector)
        pred_move = self.model.predict(self.x_valid)

        hit_counter = 0
        for i in range(len(pred_move)):
            choice = np.random.choice((0, 1), 1, p=pred_move[i])
            if choice[0] == self.y_valid[i][1]:
                hit_counter += 1

        return len(self.x_valid) - hit_counter # gyakorlatilag a hibák száma -> ezt kell 0-ra vinni

    def train_one(self, indiv, is_bp):
        fitness_val = None
        # Only backpropagation
        if indiv is None and is_bp is True:
            print("---- ONLY BACKPROPAGATION ----")
            sleep(2)
            # itt nem kell indiv a tflearn saját inicializálása tanul
            self.model.fit(self.x_train, self.y_train, n_epoch=self.n_epoch,
                           validation_set=(self.x_valid, self.y_valid),
                           batch_size=self.b_size, show_metric=False, snapshot_epoch=False, callbacks=self.LFcb)

            fitness_val = self.LFcb.fit_val

        # Full
        if indiv is not None and is_bp is True:
            assign_vector = self.create_assign_vector(indiv)
            self.assign_params(assign_vector)
            self.model.fit(self.x_train, self.y_train, n_epoch=self.n_epoch,
                           validation_set=(self.x_valid, self.y_valid),
                           batch_size=self.b_size, show_metric=False, snapshot_epoch=False, callbacks=self.LFcb)

            fitness_val = self.LFcb.fit_val

        # Only Differential Evolution
        if indiv is not None and is_bp is False:
            # assign_vector = self.create_assign_vector(indiv)
            # self.assign_params(assign_vector)
            fitness_val = self.back_t_fit(indiv)

        return fitness_val

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