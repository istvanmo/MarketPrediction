# Implementation of the following article:
# http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155133#sec014
# Here I used Differential Evolution instead of a real valued GA

from scipy.optimize import differential_evolution
import numpy as np
import ANN as ann
from time import sleep

# Parameters
only_bp = False
is_bp = True  # If only_bf False and is_bp is True -> DE + BP

h_layer_num = 10
l_rate = 0.01
momentum = 0.4
n_epoch = 10
batch_size = 128
valid_rate = 0.1717  # 0.1717 if BP True
dna_size = 9 * h_layer_num + h_layer_num + h_layer_num + 2
# [(9, 10), (10,), (10, 2), (2,)]
bounds = [(-0.1, 0.1)] * (9 * h_layer_num) + [(-0.5, 0.5)] * h_layer_num + [(-0.1, 0.1)] * (2 * h_layer_num) + [(-0.5, 0.5)] * 2
# bounds = [(-1, 1)] * dna_size

bp = ann.BP_ANN(h_layer_num, l_rate, momentum, n_epoch, batch_size, valid_rate, opt="nesterov", do=only_bp)

if only_bp:
    fit = bp.train_one(None, is_bp=only_bp)
    print("Global loss at the end of the training: ", fit)
else:
    print("---- DIFFERENTIAL EVOLUTION IS ON ----")
    if is_bp:
        print("---- BACKPROPAGATION IS ON ----")
    sleep(2)
    # Fitness function of the DE
    def BP_fit_fun(x):
        x_ravel = np.ravel(x)
        fit = bp.train_one(np.array(x_ravel), is_bp=is_bp)  # is_bp=False -> only DE and the fitness is the hit ration on the validation set
        return fit

    result = differential_evolution(BP_fit_fun, bounds, maxiter=10, popsize=15, disp=True, tol=0.000001)

    print(result.message)
    print(result.x)
    print("Number of test points: ", len(bp.y_valid))
    wrong = bp.back_t_fit(result.x)
    print("Wrong: ", wrong)
    print("Hit ratio: ", (len(bp.y_valid) - wrong) / len(bp.y_valid))
