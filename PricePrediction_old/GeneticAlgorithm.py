import numpy as np
from geneticalgs import RealGA





def func(val):
    return (val[0] + 2*val[1] - 7)**2 + (2*val[0] + val[1] - 5)**2

# GA standard settings
generation_num = 100
population_size = 2
elitism = True
selection = 'rank'
tournament_size = None # in case of tournament selection
mut_type = 1
mut_prob = 0.05
cross_type = 1
cross_prob = 0.95
optim = 'min' # minimize or maximize a fitness value? May be 'min' or 'max'.
interval = (-10, 10)

sga = RealGA(func, optim=optim, elitism=elitism, selection=selection,
            mut_type=mut_type, mut_prob=mut_prob,
            cross_type=cross_type, cross_prob=cross_prob)

sga.init_random_population(population_size, 2, interval)

fitness_progress = sga.run(generation_num)
print(sga.best_solution)







#from time import sleep
#
#
#class GA:
#    def __init__(self, dna_size, pop_size, cross_rate, mutation_rate):
#        self.dna_size = dna_size
#        self.pop_size = pop_size
#        self.cross_rate = cross_rate
#        self.mutation_rate = mutation_rate
#        # self.n_generation = n_generations
#        self.pop = np.random.uniform(-1, 1, size=(pop_size, dna_size))  # initialize the pop DNA
#
#    def select(self, pop, fitness):    # nature selection wrt pop's fitness
#        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
#                               p=fitness/np.sum(np.array(fitness)))
#        return pop[idx]
#
#    def crossover(self, parent, pop):     # mating process (genes crossover)
#        if np.random.rand() < self.cross_rate:
#            i_ = np.random.randint(0, self.pop_size, size=1)                # select another individual from pop
#            cross_points = np.random.randint(0, 2, size=self.dna_size).astype(np.bool)   # choose crossover points
#            parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
#        return parent
#
#    def mutate(self, child):
#        for point in range(self.dna_size):
#            if np.random.rand() < self.mutation_rate:
#                child[point] = 1 if child[point] == 0 else 0
#        return child
#
#    def return_pop(self):
#        return self.pop
#
#    def evolution(self, fitness):
#        self.pop = self.select(self.pop, fitness)
#        pop_copy = self.pop.copy()
#        for parent in self.pop:
#            child = self.crossover(parent, pop_copy)
#            child = self.mutate(child)
#            parent[:] = child  # parent is replaced by its child


#    def pop_diver(self):
#        for i, dna in enumerate(pop):
#
