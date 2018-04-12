from scipy.optimize import differential_evolution
import numpy as np
import ANN as ann

# Parameters
f_layer_num = 10
l_rate = 0.005
momentum = 0.4
n_epoch = 10
batch_size = 32
valid_rate = 0.6  # 0.1717
dna_size = 9 * f_layer_num + f_layer_num + f_layer_num + 1
bounds = [(-0.1, 0.1)] * dna_size

bp = ann.BP_ANN(f_layer_num, l_rate, momentum, n_epoch, batch_size, valid_rate)


def fit_fun(x):
    x_ravel = np.ravel(x)
    fit = bp.train_one(np.array(x_ravel), False)
    return fit

result = differential_evolution(fit_fun, bounds, maxiter=3000, popsize=25, disp=True, tol=0.000001)
print(result.x)
print("Próbapontok száma: ", len(bp.y_valid))
print("Helytelenek száma: ", result.fun)
print("Helyesek aránya: ", (len(bp.y_valid) - result.fun)/len(bp.y_valid))
print(result.message)