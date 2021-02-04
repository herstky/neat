class Connection:
    def __init__(self, input_node, output_node, weight, recurrent=False):
        self.input_node = input_node
        self.input_node.output_connections.append(self)
        self.output_node = output_node
        self.output_node.input_connections.append(self)
        self.weight = weight
        self.recurrent = recurrent
        self.input = 0
    

    @property
    def output(self):
        return self.input * self.weight


