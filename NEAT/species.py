import numpy as np
from NEAT.genome import Genome


class Species:
    """
    Represents a species in NEAT. Each species contains a group of members
    (individuals) and has a representative genome for comparison purposes.

    Attributes:
        id (int): The unique identifier for the species.
        members (list): A list of members (individuals) belonging to the species.
        representative: The representative genome of the species, used for comparison.
    """

    def __init__(self, id: int):
        """
        Initializes a new Species instance.
        Args:
            id (int): The unique identifier for the species.
        """
        self.id = id
        self.members = []
        self.representative = None

    @property
    def average_fitness(self):
        """
        Calculates the average fitness of the members in the species.

        Returns:
            float: The average fitness of the members, or negative infinity if there are no members.
        """
        return (
            np.mean([member.fitness for member in self.members])
            if len(self.members) > 0
            else -float("inf")
        )

    @property
    def average_adjusted_fitness(self):
        """
        Calculates the average adjusted fitness of the members in the species.

        Returns:
            float: The average adjusted fitness of the members, or negative infinity if there are no members.
        """
        return (
            np.mean([member.adjusted_fitness for member in self.members])
            if len(self.members) > 0
            else -float("inf")
        )

    @property
    def total_adjusted_fitness(self):
        """
        Calculates the total adjusted fitness of the members in the species.

        Returns:
            float: The sum of the adjusted fitness values of all members, or 0.0 if there are no members.
        """
        return (
            sum(member.adjusted_fitness for member in self.members)
            if len(self.members) > 0
            else 0.0
        )

    def __len__(self):
        """
        Returns the number of members in the species.

        Returns:
            int: The number of members in the species.
        """
        return len(self.members)


def genetic_distance(
    genome1: Genome, genome2: Genome, c1: float = 1.0, c2: float = 1.0, c3: float = 1.0
):
    """
    Calculates the genetic distance between two genomes based on their connections,
    node activations, and weight differences.

    Args:
        genome1: The first genome object, containing connection and node information.
        genome2: The second genome object, containing connection and node information.
        c1 (float, optional): Coefficient for the disjoint genes term. Defaults to 1.0.
        c2 (float, optional): Coefficient for the uncommon activations term. Defaults to 1.0.
        c3 (float, optional): Coefficient for the average weight difference term. Defaults to 1.0.

    Returns:
        float: The calculated genetic distance between the two genomes.
    """
    #c1 = len(genome1.input_node_ids) * len(genome1.output_node_ids)
    # get the genes.
    genes1 = set(genome1.connections.keys())
    genes2 = set(genome2.connections.keys())

    # find the disjoint genes.
    disjoint_genes = genes1 ^ genes2
    # find the common genes.
    common_genes = genes1 & genes2
    n_common_genes = len(common_genes) if len(common_genes) > 0 else 1

    # compute the average weight differences of matching genes.
    avg_weight_diff = (
        sum(
            abs(
                genome1.connections[conn_id].weight
                - genome2.connections[conn_id].weight
            )
            for conn_id in common_genes
        )
        / n_common_genes
    )

    # counting different activations
    uncommon_activations = 0
    n_genes1 = set(genome1.nodes.keys())
    n_genes2 = set(genome2.nodes.keys())
    common_n_genes = n_genes1 & n_genes2
    for node in common_n_genes:
        if genome1.nodes[node].activation != genome2.nodes[node].activation:
            uncommon_activations += 1

    # normalizing factor for genome size
    N = max(len(genes1), len(genes2))
    N = N if N > 0 else 1
    #N = 1 if N < (len(genome1.input_node_ids) * len(genome1.output_node_ids)) * 1.2 else N

    # compute the genetic distance
    distance = (
        ((c1 * len(disjoint_genes)) / N)
        + (c2 * uncommon_activations / len(common_n_genes))
        + (c3 * avg_weight_diff)
    )

    return distance
