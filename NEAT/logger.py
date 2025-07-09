import numpy as np
from NEAT.genome import Genome
from NEAT.species import Species


class Logger:
    """
    A utility class for logging information about the progress of a neuroevolution algorithm.
    """

    @staticmethod
    def log(generation: int, winner: Genome, species: list[Species]):
        """
        Logs the current generation, the fitness of the winning genome, and statistics about each species.

        Args:
            generation (int): The current generation number.
            winner (object): The genome with the highest fitness in the current generation.
            It is expected to have a `fitness` attribute.
            species (list[Species]): A list of species objects.
        """
        print(
            "Generation "
            + str(generation)
            + "  -  Fit: "
            + str(np.round(winner.fitness, decimals=2))
        )

        print("| spec | #mem | avg fit | best fit | best shape |")
        print("|------|------|---------|----------|------------|")
        for s in species:
            line = "| "
            line += str(s.id)
            for _ in range(7 - len(line)):
                line += " "
            line += "| "
            line += str(len(s))
            for _ in range(14 - len(line)):
                line += " "
            line += "| "
            line += str(np.round(s.average_fitness, decimals=1))
            for _ in range(24 - len(line)):
                line += " "
            line += "| "
            line += str(np.round(max([m.fitness for m in s.members]), decimals=1))
            for _ in range(35 - len(line)):
                line += " "
            line += "| "
            species_winner = max(s.members, key=lambda m: m.fitness)
            shape = (
                len(species_winner.nodes),
                len(
                    [
                        c.connection
                        for c in species_winner.connections.values()
                        if c.enabled
                    ]
                ),
            )
            line += str(shape)
            for _ in range(48 - len(line)):
                line += " "
            line += "|"
            print(line)

        print("'------'------'---------'----------'------------'\n")
