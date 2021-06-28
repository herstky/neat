from kypy_neat.utils.math import sigmoid
from kypy_neat.genes import NodeType

class Trait:
    def __init__(self, gene):
        self._gene = gene

    @property
    def gene(self):
        return self._gene

class Node(Trait):
    def __init__(self, gene):
        super().__init__(gene)
        self.input_connections = []
        self.output_connections = []
        self._output = 0
        self.aggregate_input = 0
        self.activation_count = 0
        self.activation = 0
        self.prev_activation = 0
        self.active = False
        self._stability_threshold = 1E-9

    @property
    def node_type(self):
        return self.gene.node_type

    @property
    def stable(self):
        return abs(self.activation - self.prev_activation) <= self._stability_threshold

    def activate(self):
        self.activation_count += 1
        self.prev_activation = self.activation
        self.activation = self.activation_function(self.aggregate_input)

    def activation_function(self, val):
        return sigmoid(val, 4.9)

    def flush_back(self):
        if self.node_type is NodeType.INPUT:
            self.activation = 0
            self.activation_count = 0
        else:
            if self.activation_count > 0:
                self.activation = 0
                self.activation_count = 0
            
            for conn in self.input_connections:
                if conn.input_node.activation_count > 0:
                    conn.input_node.flush_back()

class Connection(Trait):
    def __init__(self, gene, input_node, output_node):
        super().__init__(gene)
        self.input_node = input_node
        self.input_node.output_connections.append(self)
        self.output_node = output_node
        self.output_node.input_connections.append(self)

    @property
    def weight(self):
        return self.gene.weight

    @property
    def enabled(self):
        return self.gene.enabled


