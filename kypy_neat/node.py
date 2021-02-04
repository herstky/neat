from kypy_neat.utils.math import sigmoid


class Node:
    def __init__(self, rcc=False):
        self.rcc = rcc
        self.input_connections = []
        self.output_connections = []
        self._output = 0
        self.location = 0

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
