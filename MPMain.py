import GeneticAlgorithm as genalg
import ANN as ann

# BP-ANN
f_layer_num = 10
l_rate = 0.001
momentum = 0.4
n_epoch = 1
batch_size = 32
valid_rate = 0.02

# GA
dna_size = 9 * f_layer_num + f_layer_num + f_layer_num + 1
pop_size = 2
cross_rate = 0.7
mutation_rate = 0.04
n_generation = 2


ga = genalg.GA(dna_size, pop_size, cross_rate, mutation_rate)
bp = ann.BP_ANN(f_layer_num, l_rate, momentum, n_epoch, batch_size, valid_rate)

for i in range(n_generation):
    actual_pop = ga.return_pop()
    actual_fitnesses = bp.train_all(actual_pop)
