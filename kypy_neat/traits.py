from kypy_neat.utils.math import sigmoid

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
        self.location = 0

    @property
    def rcc(self):
        return self.gene.rcc

    def calculate_output(self):
        total_input = 0
        for con in self.input_connections:
            total_input += con.output

        self._output = self.activation_function(total_input)

    def output(self):
        for con in self.output_connections:
            con.input = self._output

    def activation_function(self, val):
        return sigmoid(val)

class Connection(Trait):
    def __init__(self, gene, input_node, output_node):
        super().__init__(gene)
        self.input_node = input_node
        self.input_node.output_connections.append(self)
        self.output_node = output_node
        self.output_node.input_connections.append(self)
        self.input = 0

    def _build(self):
        pass

    @property
    def weight(self):
        return self.gene.weight

    @property
    def recurrent(self):
        return self.gene.recurrent

    @property
    def output(self):
        return self.input * self.weight


