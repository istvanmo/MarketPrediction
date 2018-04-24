from geneticalgs import RealGA
import numpy as np
import ANN as ann
from time import sleep

# Training options
only_bp = True
is_bp = False  # If only_bf False and is_bp is True -> DE + BP

# Parameters
h_layer_num = 32  # for both of the hidden layers
l_rate = 0.1
momentum = 0.4
n_epoch = 3000
batch_size = 16
test_rate = 0.214  # 0.214 if BP True
dna_size = 9 * h_layer_num + h_layer_num + h_layer_num * 2 + 2

# [(9, 10), (10,), (10, 2), (2,)]
# bounds = [(-0.1, 0.1)] * (9 * h_layer_num) + [(-0.5, 0.5)] * h_layer_num + [(-0.1, 0.1)] * (2 * h_layer_num) + [(-0.5, 0.5)] * 2
# bounds = [(-1, 1)] * dna_size

bp = ann.BP_ANN(h_layer_num, l_rate, momentum, n_epoch, batch_size, test_rate, opt="momentum", do=True)

if only_bp:
    fit = bp.train_one(None, is_bp=only_bp)
    print("Loss and accuracy on the test data: ", fit)
else:
    print("---- GENETIC ALGORITHM IS ON ----")
    if is_bp:
        print("---- BACKPROPAGATION IS ON ----")
    sleep(2)
    # Fitness function of the GA
    def BP_fit_fun(x):
        x_ravel = np.ravel(x)
        # fit is loss and accuracy in a list
        fit = bp.train_one(np.array(x_ravel), is_bp=is_bp)  # is_bp=False -> only DE and the fitness is the hit ration on the validation set
        return fit[0]

    ga_opt = "min" if is_bp else "max"
    gen_alg = RealGA(BP_fit_fun, optim=ga_opt, cross_prob=0.7, mut_prob=0.2)
    # gen_alg.init_population(bounds)
    gen_alg.init_random_population(20, dna_size, (-0.5, 0.5))
    gen_alg.run(300)

    best_indiv, score = gen_alg.best_solution
    print("Number of test points: ", len(bp.y_test))
    right = bp.back_t_fit(best_indiv, for_train=False)
    print("Wrong: ", right[0])
    print("Hit ratio: ", right[0] / len(bp.y_test))
