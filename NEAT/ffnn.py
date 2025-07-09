####################################################################################################################
# NOTE
# Contents of this file are from Code Reclaimers "neat-python" and are slightly adjusted to fit this projects needs.
# Refer to: https://github.com/CodeReclaimers/neat-python
# At the end of this file, the activation functions are defined.
####################################################################################################################
import math

from NEAT.genome import Genome


class FeedForwardNetwork(object):
    """
    A class for creating a feed-forward neural network.
    Use FeedForwardNetwork.create(genome) to create a network from a genome.
    The network is activated with the activate() method, which takes a list as input.
    """

    def __init__(self, inputs: list, outputs: list, node_evals: list):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs: list):
        """
        Activate the network with the given inputs.

        Args:
            inputs (list): The inputs to the network.

        Raises:
            RuntimeError: In case the number of inputs does not match the number of input nodes.

        Returns:
            list: The outputs of the network.
        """
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(
                "Expected {0:n} inputs, got {1:n}".format(
                    len(self.input_nodes), len(inputs)
                )
            )

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, bias, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = sum(node_inputs)
            self.values[node] = activation_functions[act_func](bias + s)

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def required_for_output(inputs: list, outputs: list, connections: list):
        """
        Collect the nodes whose state is required to compute the final network output(s).

        Args:
            inputs (list): List of the input node identifiers.
            outputs (list): List of the output node identifiers.
            connections (list): List of (input, output) connections in the network.

        Note:
            It is assumed that the input identifier set and the node identifier set are disjoint.
            By convention, the output node identifiers are always the same as the output index.

        Returns:
            set: A set of identifiers of required nodes.
        """
        assert not set(inputs).intersection(outputs)

        required = set(outputs)
        s = set(outputs)
        while 1:
            # Find nodes not in s whose output is consumed by a node in s.
            t = set(a for (a, b) in connections if b in s and a not in s)

            if not t:
                break

            layer_nodes = set(x for x in t if x not in inputs)
            if not layer_nodes:
                break

            required = required.union(layer_nodes)
            s = s.union(t)

        return required

    @staticmethod
    def feed_forward_layers(inputs: list, outputs: list, connections: list):
        """
        Collect the layers whose members can be evaluated in parallel in a feed-forward network.

        Args:
            inputs (list): List of the network input nodes.
            outputs (list): List of the output node identifiers.
            connections (list): List of (input, output) connections in the network.

        Returns:
            list: A list of layers, with each layer consisting of a set of node identifiers.
              Note that the returned layers do not contain nodes whose output is ultimately
              never used to compute the final network output.
        """
        required = FeedForwardNetwork.required_for_output(inputs, outputs, connections)

        layers = []
        s = set(inputs)
        while 1:
            # Find candidate nodes c for the next layer. These nodes should connect
            # a node in s to a node not in s.
            c = set(b for (a, b) in connections if a in s and b not in s)
            # Keep only the used nodes whose entire input set is contained in s.
            t = set()
            for n in c:
                if n in required and all(a in s for (a, b) in connections if b == n):
                    t.add(n)

            if not t:
                break

            layers.append(t)
            s = s.union(t)

        return layers

    @staticmethod
    def create(genome: Genome):
        """
        Create a feed-forward network from a genome.

        Args:
            genome (Genome): The genome to create the network from (it's phenotype).

        Returns:
            FeedForwardNetwork: The genomes feed-forward network.
        """
        # Gather expressed connections
        connections = [
            cg.connection for cg in genome.connections.values() if cg.enabled
        ]

        layers = FeedForwardNetwork.feed_forward_layers(
            genome.input_node_ids, genome.output_node_ids, connections
        )
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))

                ng = genome.nodes[node]
                node_evals.append((node, ng.activation, ng.bias, inputs))

        return FeedForwardNetwork(
            genome.input_node_ids, genome.output_node_ids, node_evals
        )


##########################################################################
# The activation functions used in the network.
# Add any further activation functions here.
##########################################################################

activation_functions = {
    "sigmoid": lambda x: sigmoid_activation(x),
    "relu": lambda x: relu_activation,
}


def sigmoid_activation(z):
    z = max(-100.0, min(100.0, 5.0 * z))  # clamping
    return 1.0 / (1.0 + math.exp(-z))


def relu_activation(z):
    z = max(-100.0, min(100.0, z))  # clamping
    return max(0, z)
