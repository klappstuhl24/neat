from math import ceil
from multiprocessing import Pool, cpu_count

from NEAT.genome import Genome


class EliteSelection:
    """
    A class to perform elite selection in NEAT. This selection method chooses the
    top-performing individuals (elites) from a population based on their fitness.

    Attributes:
        fitness_function (callable): A function that calculates the fitness of a genome.
        elite_ratio (float): The ratio of elites to select from the population. Default is 0.05.
    """

    def __init__(self, fitness_function: callable, elite_ratio: float = None):
        """
        Initializes the EliteSelection class with a fitness function.

        Args:
            fitness_function (callable): A function that takes a genome as input and returns its fitness value.
            elite_ratio (float, optional): The ratio of elites to select from the population. Defaults to 0.05.
        """
        self.fitness_function = fitness_function
        self.elite_ratio = elite_ratio if elite_ratio is not None else 0.05

    def select(self, population: list[Genome], parallel: bool = False):
        """
        Selects the top-performing individuals (elites) from the population based on their fitness.

        Args:
            population (list[Genome]): A list of genomes representing the population.
            parallel (bool, optional): Whether to calculate fitness in parallel using multiprocessing.
                                       Defaults to True.

        Returns:
            list: A list of the top-performing genomes (elites) from the population.
        """
        n_elites = ceil(self.elite_ratio * len(population)) + 1
        # check if population is too small
        if n_elites >= len(population):
            return population

        if parallel:
            # calculating fitness in parallel
            pool = Pool(processes=cpu_count() - 1 or 1)
            jobs = []

            for genome in population:
                jobs.append(pool.apply_async(self.fitness_function, (genome,)))

            for genome, job in zip(population, jobs):
                genome.fitness = job.get(timeout=None)

            pool.close()
            pool.join()
            pool.terminate()
        else:
            # calculating fitness in serial
            for genome in population:
                genome.fitness = self.fitness_function(genome)

        sorted_population = sorted(
            population, key=lambda genome: genome.fitness, reverse=True
        )
        return sorted_population[:n_elites]
