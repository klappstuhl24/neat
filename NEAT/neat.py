import random

from NEAT.genome import Genome
from NEAT.logger import Logger
from NEAT.species import *


class NEAT:
    """
    A class implementing the NeuroEvolution of Augmenting Topologies (NEAT) algorithm.
    Attributes:
        winner (Genome): The genome with the highest fitness in the population.
        selection (Selection): The selection strategy used to choose genomes for reproduction.
        crossover (Crossover): The crossover strategy used to combine genomes.
        mutation (Mutation): The mutation strategy used to modify genomes.
        population (list): The current population of genomes.
        species (list): The list of species in the population.
        species_id_counter (int): Counter to assign unique IDs to species.
        distance_threshold (float): The threshold for determining if two genomes belong to the same species.
        pop_size (int): The size of the population.
    """

    def __init__(
        self,
        selection,
        crossover,
        mutation,
        distance_threshold=3.0,
        parallel: bool = False,
    ):
        """
        Initializes the NEAT algorithm with the given selection, crossover, and mutation strategies.
        Args:
            selection (Selection): The selection strategy.
            crossover (Crossover): The crossover strategy.
            mutation (Mutation): The mutation strategy.
            distance_threshold (float, optional): The threshold for species differentiation. Defaults to 3.0.
        """
        self.winner = None
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.population = []
        self.species = []
        self.species_id_counter = 0  # to keep track of species ids
        self.distance_threshold = distance_threshold
        self.parallel = parallel

    def speciate(self):
        """
        Groups genomes into species based on their genetic distance and adjusts their fitness values.
        This method:
        - Clears previous species while retaining a random representative.
        - Assigns genomes to existing species or creates new species if no match is found.
        - Removes empty species.
        - Adjusts the fitness of genomes based on the size of their species.
        """
        # clear previous species but keep a random representative
        for species in self.species:
            species.representative = random.choice(species.members)
            species.members.clear()

        for genome in self.population:
            # check if genome fits into an existing species
            placed = False
            for species in self.species:
                if (
                    genetic_distance(genome, species.representative)
                    < self.distance_threshold
                ):
                    species.members.append(genome)
                    placed = True
                    break
            # if no suitable species is found, create a new one
            if not placed:
                self.species_id_counter += 1
                new_species = Species(self.species_id_counter)
                new_species.members.append(genome)
                new_species.representative = genome
                self.species.append(new_species)
        # remove empty species
        for species in self.species[:]:
            if len(species) == 0:
                self.species.remove(species)
        # set adjusted fitness for each genome
        for species in self.species:
            for genome in species.members:
                genome.adjusted_fitness = genome.fitness / len(species)

    def reproduce(self):
        """
        Reproduces the next generation of genomes by applying selection, crossover, and mutation.
        This method:
        - Clears the current population.
        - Calculates the breeding size for each species based on adjusted fitness.
        - Selects genomes for reproduction and generates offspring through crossover and mutation.
        - Populates the next generation with selected genomes and offspring.
        """
        self.population.clear()
        # apply a shift in fitness to avoid negative values. This is required for the species sizes.
        fitness_shift = (
            min(m.adjusted_fitness for species in self.species for m in species.members)
            - 1
        )
        total_adjusted_fitness = sum(
            species.total_adjusted_fitness - fitness_shift * len(species)
            for species in self.species
        )

        # calculate the breeding size for each species
        breed_sizes = []
        for species in self.species:
            breed_size = round(
                (
                    (species.total_adjusted_fitness - fitness_shift * len(species))
                    / total_adjusted_fitness
                )
                * self.pop_size
            )
            breed_sizes.append(breed_size)

        for species, breed_size in zip(self.species, breed_sizes):
            # selection
            selected = self.selection.select(species.members, self.parallel)

            for i in range(breed_size):
                if i < len((selected)):
                    self.population.append(selected[i])
                else:
                    # pick two random representatives
                    parent1 = random.choice(selected)
                    parent2 = random.choice(selected)
                    # crossover
                    child = self.crossover.crossover(parent1, parent2)
                    # mutation
                    self.mutation.mutate(child)
                    # add child to population
                    self.population.append(child)

    def start(
        self,
        pop_size: int,
        genome_shape: tuple,
        generations: int,
        threshold: float = None,
        initial_connections: int = 0,
        callback: callable = None,
    ):
        """
        Starts the NEAT algorithm and evolves the population over a specified number of generations.

        Args:
            pop_size (int): The size of the population.
            genome_shape (tuple): The shape of the genome (input nodes, output nodes).
            generations (int): The number of generations to evolve.
            threshold (float, optional): The fitness threshold to stop evolution early. Defaults to None.
            initial_connections (int, optional): The number of initial connections in each genome. Defaults to 0.
        Returns:
            Genome: The genome with the highest fitness after evolution.
        """
        # fill population
        self.pop_size = pop_size
        self.population.clear()
        for _ in range(self.pop_size):
            self.population.append(
                Genome(genome_shape[0], genome_shape[1], initial_connections)
            )
        # calculate fitness values for the first time
        self.selection.select(self.population, self.parallel)

        for g in range(generations):
            # create species
            self.speciate()
            # calculate best fitness
            self.winner = self.population[0]
            for genome in self.population:
                if genome.fitness > self.winner.fitness:
                    self.winner = genome

            # check if threshold is reached
            if threshold is not None:
                if self.winner.fitness >= threshold:
                    # log final progress
                    Logger.log(g + 1, self.winner, self.species)
                    break

            # print the current state
            Logger.log(g + 1, self.winner, self.species)
            # callback function
            if callback != None:
                callback()
            self.reproduce()

        print("Done!")
        return self.winner
