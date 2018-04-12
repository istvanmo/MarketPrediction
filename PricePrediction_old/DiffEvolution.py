# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   A simple, bare bones, implementation of differential evolution with Python
#   August, 2017
#
# ------------------------------------------------------------------------------+
# Modified version

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

import random
import ANN as ann
import numpy as np


# GANYÉ
f_layer_num = 10
l_rate = 0.001
momentum = 0.4
n_epoch = 10
batch_size = 128
valid_rate = 0.1717
dna_size = 9 * f_layer_num + f_layer_num + f_layer_num + 1

bp = ann.BP_ANN(f_layer_num, l_rate, momentum, n_epoch, batch_size, valid_rate)


# --- FUNCTIONS ----------------------------------------------------------------+


def ensure_bounds(vec, bounds):
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])

    return vec_new


# --- MAIN ---------------------------------------------------------------------+

def main(bounds, popsize, mutate, recombination, maxiter):
    # --- INITIALIZE A POPULATION (step #1) ----------------+

    population = []
    for i in range(0, popsize):
        indv = []
        for j in range(len(bounds)):
            indv.append(random.uniform(bounds[j][0], bounds[j][1]))
        population.append(indv)

    # --- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    for i in range(1, maxiter + 1):
        print('GENERATION:', i)

        gen_scores = []  # score keeping

        # cycle through each individual in the population
        for j in range(0, popsize):

            # --- MUTATION (step #3.A) ---------------------+

            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = list(range(0, popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 3)

            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]  # target individual

            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor, bounds)

            # --- RECOMBINATION (step #3.B) ----------------+

            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])

                else:
                    v_trial.append(x_t[k])

            # --- GREEDY SELECTION (step #3.C) -------------+
            # általam eleje
            v_trial_np_array = np.ravel(np.array(v_trial))
            x_t_np_array = np.ravel(np.array(x_t))
            # általam vége
            score_trial = bp.train_one(v_trial_np_array, False)
            score_target = bp.train_one(x_t_np_array, False)

            if score_trial < score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)
                print('   >', score_trial, v_trial)

            else:
                print('   >', score_target, x_t)
                gen_scores.append(score_target)

        # --- SCORE KEEPING --------------------------------+

        gen_avg = sum(gen_scores) / popsize  # current generation avg. fitness
        gen_best = min(gen_scores)  # fitness of best individual
        gen_sol = population[gen_scores.index(min(gen_scores))]  # solution of best individual

        print('      > GENERATION AVERAGE:', gen_avg)
        print('      > GENERATION BEST:', gen_best)
        print('         > BEST SOLUTION:', gen_sol, '\n')

    return gen_sol


# --- CONSTANTS ----------------------------------------------------------------+

bounds = [(-0.5, 0.5)] * dna_size # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
popsize = 100  # Population size, must be >= 4
mutate = 0.8  # Mutation factor [0,2]
recombination = 0.7  # Recombination rate [0,1]
maxiter = 1000  # Max number of generations (maxiter)

# --- RUN ----------------------------------------------------------------------+

main(bounds, popsize, mutate, recombination, maxiter)

# --- END ----------------------------------------------------------------------+
