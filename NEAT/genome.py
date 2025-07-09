import random
from enum import Enum, auto


class NodeType(Enum):
    """
    Enum for node types in the genome.
    """

    SENSOR = auto()
    OUTPUT = auto()
    HIDDEN = auto()


class NodeGene:
    """
    Represents a node in a neural network genome for NEAT.

    Attributes:
        id (int): A unique identifier for the node.
        type (str): The type of the node (e.g., 'input', 'hidden', 'output').
        bias (float): The bias value associated with the node.
        activation (str): The activation function used by the node.
    """

    def __init__(self, id: int, type: NodeType, bias: float, activation: str):
        self.id = id
        self.type = type
        self.bias = bias
        self.activation = activation

    def __str__(self):
        return (
            "NODE "
            + str(self.id)
            + "\n"
            + str(self.type.value)
            + "\n"
            + "Bias: "
            + str(self.bias)
        )


class ConnectionGene:
    """
    Represents a connection gene in a neural network genome for NEAT.

    A connection gene defines a directed connection between two nodes in the network,
    along with its associated weight and whether the connection is enabled.

    Attributes:
        id (int): A unique identifier for the connection gene.
        connection (tuple): A tuple (in_node, out_node) representing the input and output nodes of the connection.
        weight (float): The weight of the connection.
        enabled (bool): A flag indicating whether the connection is enabled or disabled.
    """

    def __init__(
        self, id: int, in_node: int, out_node: int, weight: float, enabled: bool
    ):
        self.id = id
        self.connection = (in_node, out_node)
        self.weight = weight
        self.enabled = enabled

    def __str__(self):
        return (
            'IN: ' + str(self.connection[0]) + '\n' +
            'OUT: ' + str(self.connection[1]) + '\n' +
            'WEIGHT: ' + str(self.weight) + '\n' +
            'ENABLED: ' + str(self.enabled) + '\n' +
            'INNOV: ' + str(self.id)
        )



class Genome:
    """
    Represents a genome in the NEAT algorithm.
    A genome consists of nodes and connections that define a neural network.

    Attributes:
        nodes (dict): A dictionary of NodeGene objects, where keys are node IDs.
        connections (dict): A dictionary of ConnectionGene objects, where keys are tuples of (input_node_id, output_node_id).
        input_node_ids (list): A list of IDs for input nodes. Negative by default.
        output_node_ids (list): A list of IDs for output nodes. 0 to n by default.
        fitness (float): The fitness score of the genome.
        adjusted_fitness (float): The adjusted fitness score of the genome.
    """

    def __init__(self, n_inputs: int, n_outputs: int, initial_connections: int = -1):
        """
        Initializes a Genome object with the specified number of input and output nodes.

        Args:
            n_inputs (int): The number of input nodes.
            n_outputs (int): The number of output nodes.
            initial_connections (int, optional): The number of initial random connections.
                If set to -1, it defaults to the maximum possible connections (n_inputs * n_outputs).
        """
        self.nodes = {}  # dictionary of NodeGene objects
        self.connections = {}  # dictionary of ConnectionGene objects
        self.input_node_ids = []
        self.output_node_ids = []
        self.fitness = 0.0
        self.adjusted_fitness = 0.0

        for i in range(n_inputs):
            # apply negative id for input nodes
            id = -(i + 1)
            self.nodes[id] = NodeGene(
                id, NodeType.SENSOR, random.uniform(-1, 1), "sigmoid"
            )
            self.input_node_ids.append(id)

        for o in range(n_outputs):
            # apply positive ids from 0 to n for output nodes
            self.nodes[o] = NodeGene(
                o, NodeType.OUTPUT, random.uniform(-1, 1), "sigmoid"
            )
            self.output_node_ids.append(o)

        # set initial connections to max if not specified
        initial_connections = (
            initial_connections if initial_connections >= 0 else n_inputs * n_outputs
        )
        
        possible_connections = []
        for in_node in self.input_node_ids:
            for out_node in self.output_node_ids:
                possible_connections.append((in_node, out_node))
                
        for i in range(min(initial_connections, n_inputs * n_outputs)):
            # add new random connections
            connection_id = random.choice(possible_connections)
            possible_connections.remove(connection_id)
            self.connections[connection_id] = ConnectionGene(
                connection_id,
                connection_id[0],
                connection_id[1],
                random.uniform(-1, 1),
                True,
            )

    def get_new_node_id(self):
        """
        Generates a new unique node ID for the genome.

        Returns:
            int: A new node ID that is greater than the current maximum node ID.
        """
        return max(self.nodes.keys()) + 1

    def creates_cycle(self, connections: list, test: tuple):
        """
        NOTE: This function is from Code Reclaimers 'neat-python' repository and adjusted to fit the current code.
        Refer to: https://github.com/CodeReclaimers/neat-python
        Checks if adding a new connection would create a cycle in the network.

        Args:
            connections (list of tuple): A list of existing connections, where each connection is a tuple (input_node_id, output_node_id).
            test (tuple): The new connection to test, represented as a tuple (input_node_id, output_node_id).

        Returns:
            bool: True if the new connection creates a cycle, False otherwise.
        """
        i, o = test
        if i == o:
            return True

        visited = {o}
        while True:
            num_added = 0
            for a, b in connections:
                if a in visited and b not in visited:
                    if b == i:
                        return True

                    visited.add(b)
                    num_added += 1

            if num_added == 0:
                return False
