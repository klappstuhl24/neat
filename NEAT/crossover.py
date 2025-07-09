import random
from copy import deepcopy

from NEAT.genome import *


class Crossover:
    """
    A class to perform genetic crossover operations for NEAT.

    The crossover operation combines the genetic information of two parent genomes
    to produce a child genome. The fitter parent contributes more genetic material
    to the child, while ensuring compatibility and validity of the resulting genome.
    """

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """
        Perform a crossover operation between two parent genomes to produce a child genome.

        The fitter parent (higher fitness) contributes more genetic material to the child.
        Connections that exist in both parents are randomly inherited from either parent.
        Connections unique to the fitter parent are always inherited. Connections unique
        to the less fit parent are inherited only if they meet specific criteria.

        Args:
            parent1 (Genome): The first parent genome.
            parent2 (Genome): The second parent genome.

        Returns:
            Genome: A new genome object representing the child, with inherited connections
            and nodes from the parents.
        """
        if parent1.fitness < parent2.fitness:
            parent1, parent2 = parent2, parent1
        # assuming parent1.fitness >= parent2.fitness
        child = deepcopy(parent1)
        # reset connections in child
        child.connections = {}

        # iterate over all connections in the fitter parent (parent1)
        for conn_id in parent1.connections:
            # if connection exists in both parents
            if conn_id in parent2.connections:
                # randomly select which connection to inherit
                if random.random() < 0.5:
                    child.connections[conn_id] = deepcopy(parent1.connections[conn_id])
                else:
                    child.connections[conn_id] = deepcopy(parent2.connections[conn_id])

            # if the connection exists only in the fitter parent (parent1)
            else:
                child.connections[conn_id] = deepcopy(parent1.connections[conn_id])

        for conn_id in parent2.connections:
            # if the connection does not exist in the fitter parent (parent1)
            if conn_id not in parent1.connections:
                # only inherit this connection if the in_node is not an input node and the nodes exist in fitter parent (parent1)
                if (
                    parent2.connections[conn_id].connection[0] >= 0
                    and (conn_id[0] in parent1.nodes.keys())
                    and (conn_id[1] in parent1.nodes.keys())
                ):
                    child.connections[conn_id] = deepcopy(parent2.connections[conn_id])

        # make sure there are no connections that lead to an input node
        to_delete = []
        for conn in child.connections:
            if conn[1] < 0 or conn[1] in child.input_node_ids:
                to_delete.append(conn)

        for i in to_delete:
            del child.connections[i]

        return child
