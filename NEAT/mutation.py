import random

from NEAT.genome import ConnectionGene, Genome, NodeGene, NodeType


class Mutation:
    """
    A class to handle mutation operations for a genome in a NEAT (NeuroEvolution of Augmenting Topologies) algorithm.
    Attributes:
        add_node_prob (float): Probability of adding a new node during mutation.
        add_connection_prob (float): Probability of adding a new connection during mutation.
        mutate_weight_prob (float): Probability of mutating the weight of a connection.
        mutate_bias_prob (float): Probability of mutating the bias of a node.
        change_weight_prob (float): Probability of changing the weight of a connection to a new value.
        change_bias_prob (float): Probability of changing the bias of a node to a new value.
    """

    def __init__(
        self,
        add_node_prob: float = 0.1,
        add_connection_prob: float = 0.2,
        mutate_weight_prob: float = 0.8,
        mutate_bias_prob: float = 0.8,
        change_weight_prob: float = 0.05,
        change_bias_prob: float = 0.05,
    ):
        """
        Initializes the Mutation object with probabilities for various mutation operations.
        Args:
            add_node_prob (float): Probability of adding a new node. Default is 0.1.
            add_connection_prob (float): Probability of adding a new connection. Default is 0.2.
            mutate_weight_prob (float): Probability of mutating a connection's weight. Default is 0.8.
            mutate_bias_prob (float): Probability of mutating a node's bias. Default is 0.8.
            change_weight_prob (float): Probability of changing a connection's weight to a new value. Default is 0.05.
            change_bias_prob (float): Probability of changing a node's bias to a new value. Default is 0.05.
        """
        self.add_node_prob = add_node_prob
        self.add_connection_prob = add_connection_prob
        self.mutate_weight_prob = mutate_weight_prob
        self.mutate_bias_prob = mutate_bias_prob
        self.change_weight_prob = change_weight_prob
        self.change_bias_prob = change_bias_prob

    def mutate(self, genome: Genome):
        """
        Applies mutation operations to a given genome.
        Args:
            genome (Genome): The genome to be mutated.
        """
        # mutate every other genome
        for node in genome.nodes.keys():
            # mutate bias
            if random.random() < self.mutate_bias_prob:
                self._mutate_bias(genome, node)
            # change bias to a new value
            elif random.random() < self.change_bias_prob:
                self._change_bias(genome, node)

        for connection in genome.connections.keys():
            # mutate weight
            if random.random() < self.mutate_weight_prob:
                self._mutate_weight(genome, connection)
            # change weight to a new value
            elif random.random() < self.change_weight_prob:
                self._change_weight(genome, connection)
        # add node
        if random.random() < self.add_node_prob:
            self._add_node(genome)
        # add connection
        if random.random() < self.add_connection_prob:
            self._add_connection(genome)

    def _add_node(self, genome: Genome):
        """
        Adds a new node to the genome by splitting an existing connection.
        Args:
            genome (Genome): The genome to which a new node will be added.
        """
        if len(genome.connections) == 0:
            return
        # picking random connection to be split
        connection = random.choice(list(genome.connections.values()))
        connection.enabled = False
        # adding a node
        new_node_id = genome.get_new_node_id()
        new_node = NodeGene(
            new_node_id, NodeType.HIDDEN, random.uniform(-1, 1), "sigmoid"
        )
        genome.nodes[new_node_id] = new_node
        # connection to and from the node
        in_connection_id = (connection.connection[0], new_node_id)
        in_connection = ConnectionGene(
            in_connection_id, connection.connection[0], new_node_id, 1.0, True
        )
        out_connection_id = (new_node_id, connection.connection[1])
        out_connection = ConnectionGene(
            out_connection_id,
            new_node_id,
            connection.connection[1],
            connection.weight,
            True,
        )

        genome.connections[in_connection_id] = in_connection
        genome.connections[out_connection_id] = out_connection

    def _add_connection(self, genome: Genome):
        """
        Adds a new connection between two nodes in the genome.
        Args:
            genome (Genome): The genome to which a new connection will be added.
        """
        node1 = random.choice(list(genome.nodes.values()))
        node2 = random.choice(list(genome.nodes.values()))
        # check if these two are valid
        if node1.type == NodeType.SENSOR and node2.type == NodeType.SENSOR:
            return
        if node1.type == NodeType.OUTPUT and node2.type == NodeType.OUTPUT:
            return
        if node1.type == NodeType.OUTPUT and node2.type == NodeType.SENSOR:
            node1, node2 = node2, node1
        if node1.type != NodeType.SENSOR and node2.type == NodeType.SENSOR:
            return
        if genome.creates_cycle(
            [c.connection for c in genome.connections.values()], (node1.id, node2.id)
        ):
            return
        if (
            node2.id < 0
            or node2.type == NodeType.SENSOR
            or node2.id in genome.input_node_ids
        ):
            return
        if (node1.id, node2.id) in genome.connections.keys():
            genome.connections[(node1.id, node2.id)].enabled = True

        new_connection_id = (node1.id, node2.id)
        new_connection = ConnectionGene(
            new_connection_id, node1.id, node2.id, random.uniform(-2, 2), True
        )
        genome.connections[new_connection_id] = new_connection

    def _change_weight(self, genome: Genome, connection: tuple):
        """
        Changes the weight of a specific connection to a new random value.
        Args:
            genome (Genome): The genome containing the connection.
            connection (tuple): The connection identifier (input node ID, output node ID).
        """
        genome.connections[connection].weight = random.uniform(-2, 2)

    def _change_bias(self, genome: Genome, node: int):
        """
        Changes the bias of a specific node to a new random value.
        Args:
            genome (Genome): The genome containing the node.
            node (int): The node identifier.
        """
        genome.nodes[node].bias = random.uniform(-2, 2)

    def _mutate_weight(self, genome: Genome, connection: tuple):
        """
        Mutates the weight of a specific connection by adding a small random value.
        Args:
            genome (Genome): The genome containing the connection.
            connection (tuple): The connection identifier (input node ID, output node ID).
        """
        genome.connections[connection].weight += random.uniform(-0.5, 0.5) #random.normalvariate(0.0, 0.2)

    def _mutate_bias(self, genome: Genome, node: int):
        """
        Mutates the bias of a specific node by adding a small random value.
        Args:
            genome (Genome): The genome containing the node.
            node (int): The node identifier.
        """
        genome.nodes[node].bias += random.uniform(-0.5, 0.5) #random.normalvariate(0.0, 0.2)
