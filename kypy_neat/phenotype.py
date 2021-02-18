from enum import Enum

from kypy_neat.genes import NodeType, NodeGene, ConnectionGene
from kypy_neat.traits import Node, Connection

class StabilizationMethod(Enum):
    ITERATIVE = 1
    OUTPUT_DELTA = 2

class Phenotype:
    def __init__(self, genotype, stabilization_method=StabilizationMethod.ITERATIVE):
        self._genotype = genotype
        self._stabilization_method = stabilization_method
        self._input_nodes = []
        self._hidden_nodes = []
        self._output_nodes = []
        self._node_map = {}
        self._connection_map = {}
        self._iteration_count = 0
        self._iteration_limit = 30
        self._activation_abort_limit = 50
        self._build()

    @property
    def genotype(self):
        return self._genotype

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def hidden_nodes(self):
        return self._hidden_nodes

    @property
    def output_nodes(self):
        return self._output_nodes

    @property
    def inner_nodes(self):
        ''' Returns all input and hidden nodes.
        '''
        return self._input_nodes + self._hidden_nodes

    @property
    def outer_nodes(self):
        ''' Returns all hidden and output nodes.
        '''
        return self._hidden_nodes + self._output_nodes

    @property
    def all_nodes(self):
        return self._input_nodes + self._hidden_nodes + self._output_nodes        

    @property
    def all_active(self):
        ''' Returns True if all nodes have been activated at least once,
            otherwise returns False.
        '''
        for node in self.outer_nodes:
            if node.activation_count == 0:
                return False
        
        return True

    def _build(self):
        for node_gene in self._genotype.node_genes:
            node = self.generate_node(node_gene)

        for connection_gene in self._genotype.connection_genes:
            connection = self.generate_connection(connection_gene)

    def generate_node(self, gene):
        node = Node(gene)
        self._node_map[gene.innovation_id] = node
        if gene.node_type is NodeType.INPUT:
            self._input_nodes.append(node)
        elif gene.node_type is NodeType.HIDDEN:
            self._hidden_nodes.append(node)
        elif gene.node_type is NodeType.OUTPUT:
            self._output_nodes.append(node)

        return node

    def generate_connection(self, gene):
        input_node = self._node_map[gene.input_node_id]
        output_node = self._node_map[gene._output_node_id]
        connection = Connection(gene, input_node, output_node)
        self._connection_map[gene.innovation_id] = connection

        return connection

    def flush(self):
        for node in self.output_nodes:
            node.flush_back()

    def stabalized(self):
        if self._iteration_count < 1:
            return False

        if self._stabilization_method is StabilizationMethod.ITERATIVE:
            if self._iteration_count < self._iteration_limit:
                return False

        elif self._stabilization_method is StabilizationMethod.OUTPUT_DELTA:
            for node in self.outer_nodes:
                if not node.stable:
                    return False
        else:
            raise RuntimeError('Unrecognized stabilization method')

        return True
    
    def activate(self, inputs):
        # Ref: https://stackoverflow.com/questions/55569260/feedforward-algorithm-in-neat-neural-evolution-of-augmenting-topologies
        if type(inputs) is not list:
            inputs = [inputs]

        if len(inputs) != len(self.input_nodes):
            raise RuntimeError('Invalid input')

        for node, input_ in zip(self.input_nodes, inputs):
            node.activation = input_

        initial_pass = True # used to ensure at least one pass takes place
        abort_count = 0
        while initial_pass or not self.all_active:
            abort_count += 1
            if abort_count > self._activation_abort_limit:
                return 1000
                # raise RuntimeError('Activation limit exceeded.')

            for node in self.outer_nodes:
                node.aggregate_input = 0
                node.active = False
                for conn in node.enabled_input_connections:
                    if conn.input_node.active or conn.input_node.node_type is NodeType.INPUT:
                        node.active = True
                    node.aggregate_input += conn.weight * conn.input_node.activation        
            
            for node in self.outer_nodes:
                if node.active:
                    node.activate()

            initial_pass = False    

        return 0

    def evaluate_network(self, inputs):
        self._iteration_count = 0
        error = 0
        while not self.stabalized():
            error += self.activate(inputs)
            self._iteration_count += 1

        output = [node.activation for node in self.output_nodes]
        self.flush()

        return (output, error)
