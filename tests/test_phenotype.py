import unittest

from kypy_neat.phenotype import Phenotype
from kypy_neat.node import Node
from kypy_neat.connection import Connection


class TestPhenotype(unittest.TestCase):
    def setUp(self):
        self.phenotype = Phenotype()

    def test_topsorted_nodes1(self):
        self.phenotype.input_nodes = [Node() for i in range(3)]
        self.phenotype.hidden_nodes = [Node() for i in range(3)]
        self.phenotype.output_nodes = [Node() for i in range(3)]

        Connection(self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[0], 1) # 0
        Connection(self.phenotype.hidden_nodes[0], self.phenotype.output_nodes[0], 1) # 1
        Connection(self.phenotype.input_nodes[0], self.phenotype.output_nodes[1], 1) # 2
        Connection(self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[1], 1) # 3
        Connection(self.phenotype.input_nodes[1], self.phenotype.hidden_nodes[1], 1) # 4
        Connection(self.phenotype.hidden_nodes[1], self.phenotype.output_nodes[1], 1) # 5
        Connection(self.phenotype.input_nodes[2], self.phenotype.hidden_nodes[2], 1) # 6
        Connection(self.phenotype.hidden_nodes[2], self.phenotype.hidden_nodes[1], 1) # 7
        Connection(self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[2], 1) # 8

        sorted_nodes = self.phenotype.topsorted_nodes(self.phenotype.all_nodes)

        node_list = [
            self.phenotype.input_nodes[2],
            self.phenotype.hidden_nodes[2],
            self.phenotype.output_nodes[2],
            self.phenotype.input_nodes[1],
            self.phenotype.input_nodes[0],
            self.phenotype.hidden_nodes[0],
            self.phenotype.hidden_nodes[1],
            self.phenotype.output_nodes[1],
            self.phenotype.output_nodes[0]
        ]

        self.assertEquals(sorted_nodes, node_list)

    def test_topsorted_nodes2(self):
        self.phenotype.input_nodes = [Node()]
        self.phenotype.hidden_nodes = [Node() for i in range(3)]
        self.phenotype.output_nodes = [Node()]

        Connection(self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[1], 1) # 0
        Connection(self.phenotype.hidden_nodes[1], self.phenotype.hidden_nodes[0], 1) # 1
        Connection(self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[2], 1) # 2
        Connection(self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[0], 1) # 3

        sorted_nodes = self.phenotype.topsorted_nodes(self.phenotype.all_nodes)

        node_list = [
            self.phenotype.input_nodes[0],
            self.phenotype.hidden_nodes[1],
            self.phenotype.hidden_nodes[0],
            self.phenotype.hidden_nodes[2],
            self.phenotype.output_nodes[0]
        ]

        self.assertEquals(sorted_nodes, node_list)

    def test_update_locations1(self):
        self.phenotype.input_nodes = [Node()]
        self.phenotype.hidden_nodes = [Node() for i in range(3)]
        self.phenotype.output_nodes = [Node()]

        Connection(self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[1], 1) # 0
        Connection(self.phenotype.hidden_nodes[1], self.phenotype.hidden_nodes[0], 1) # 1
        Connection(self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[2], 1) # 2
        Connection(self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[0], 1) # 3

        self.phenotype.update_locations()
        
        locations = [node.location for node in self.phenotype.all_nodes]

        self.assertEquals(locations, [0, 2, 1, 3, 4])

    def test_update_locations2(self):
        self.phenotype.input_nodes = [Node() for i in range(3)]
        self.phenotype.hidden_nodes = [Node() for i in range(3)]
        self.phenotype.output_nodes = [Node() for i in range(3)]

        Connection(self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[0], 1) # 0
        Connection(self.phenotype.hidden_nodes[0], self.phenotype.output_nodes[0], 1) # 1
        Connection(self.phenotype.input_nodes[0], self.phenotype.output_nodes[1], 1) # 2
        Connection(self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[1], 1) # 3
        Connection(self.phenotype.input_nodes[1], self.phenotype.hidden_nodes[1], 1) # 4
        Connection(self.phenotype.hidden_nodes[1], self.phenotype.output_nodes[1], 1) # 5
        Connection(self.phenotype.input_nodes[2], self.phenotype.hidden_nodes[2], 1) # 6
        Connection(self.phenotype.hidden_nodes[2], self.phenotype.hidden_nodes[1], 1) # 7
        Connection(self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[2], 1) # 8

        self.phenotype.update_locations()
        
        locations = [node.location for node in self.phenotype.all_nodes]

        node_list = [
            self.phenotype.input_nodes[2],
            self.phenotype.hidden_nodes[2],
            self.phenotype.output_nodes[2],
            self.phenotype.input_nodes[1],
            self.phenotype.input_nodes[0],
            self.phenotype.hidden_nodes[0],
            self.phenotype.hidden_nodes[1],
            self.phenotype.output_nodes[1],
            self.phenotype.output_nodes[0]
        ]

        self.assertEquals(locations, [0, 0, 0, 1, 2, 1, 2, 3, 2])