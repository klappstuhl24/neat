import gymnasium as gym
import numpy as np
from NEAT.crossover import Crossover
from NEAT.mutation import Mutation
from NEAT.neat import NEAT
from NEAT.selection import EliteSelection

# POP_SIZE = 256
# N_INPUTS = 4
# N_OUPUTS = 2
# ENV = "CartPole-v1"
# THRESHOLD = 475
# N_SIMULATIONS = 5
# INITIAL_CONNS = 0

POP_SIZE = 256
N_INPUTS = 8
N_OUPUTS = 4
ENV = "LunarLander-v3"
THRESHOLD = 200
N_SIMULATIONS = 5
INITIAL_CONNS = 16

import my_fitness_function

selection = EliteSelection(my_fitness_function.fitness_function)
crossover = Crossover()
mutation = Mutation()

neat = NEAT(selection, crossover, mutation, distance_threshold=2.0)
winner = neat.start(POP_SIZE, (N_INPUTS, N_OUPUTS), 100000, THRESHOLD, INITIAL_CONNS)
