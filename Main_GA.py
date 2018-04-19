from geneticalgs import RealGA
import numpy as np
import ANN as ann
from time import sleep

# Parameters
only_bp = True
is_bp = True  # If only_bf False and is_bp is True -> DE + BP

h_layer_num = 10
l_rate = 0.001
momentum = 0.4
n_epoch = 100000
batch_size = 64
valid_rate = 0.1717  # 0.1717 if BP True
dna_size = 9 * h_layer_num + h_layer_num + h_layer_num * 2 + 2

# [(9, 10), (10,), (10, 2), (2,)]
# bounds = [(-0.1, 0.1)] * (9 * h_layer_num) + [(-0.5, 0.5)] * h_layer_num + [(-0.1, 0.1)] * (2 * h_layer_num) + [(-0.5, 0.5)] * 2
# bounds = [(-1, 1)] * dna_size

bp = ann.BP_ANN(h_layer_num, l_rate, momentum, n_epoch, batch_size, valid_rate, opt="Adam", do=only_bp)

if only_bp:
    fit = bp.train_one(None, is_bp=only_bp)
    print("Global loss at the end of the training: ", fit)
else:
    print("---- GENETIC ALGORITHM IS ON ----")
    if is_bp:
        print("---- BACKPROPAGATION IS ON ----")
    sleep(2)
    # Fitness function of the DE
    def BP_fit_fun(x):
        x_ravel = np.ravel(x)
        # fit is loss and accuracy in a list
        fit = bp.train_one(np.array(x_ravel), is_bp=is_bp)  # is_bp=False -> only DE and the fitness is the hit ration on the validation set
        return fit[0]

    gen_alg = RealGA(BP_fit_fun, optim="min", cross_prob=0.7, mut_prob=0.2)
    # gen_alg.init_population(bounds)
    gen_alg.init_random_population(50, dna_size, (-0.5, 0.5))
    gen_alg.run(100)

    best_indiv, score = gen_alg.best_solution
    print("Number of test points: ", len(bp.y_test))
    wrong = bp.back_t_fit(best_indiv)
    print("Wrong: ", wrong)
    print("Hit ratio: ", (len(bp.y_test) - wrong) / len(bp.y_test))
