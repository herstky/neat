import unittest

from kypy_neat.phenotype import Phenotype
from kypy_neat.traits import Node, Connection


class TestPhenotype(unittest.TestCase):
    def setUp(self):
        pass
#         self.phenotype = Phenotype(None)

#     def test_get_topsorted_nodes1(self):
#         self.phenotype._input_nodes = [Node(None) for i in range(3)]
#         self.phenotype._hidden_nodes = [Node(None) for i in range(3)]
#         self.phenotype._output_nodes = [Node(None) for i in range(3)]

#         Connection(None, self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[0], 1) # 0
#         Connection(None, self.phenotype.hidden_nodes[0], self.phenotype.output_nodes[0], 1) # 1
#         Connection(None, self.phenotype.input_nodes[0], self.phenotype.output_nodes[1], 1) # 2
#         Connection(None, self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[1], 1) # 3
#         Connection(None, self.phenotype.input_nodes[1], self.phenotype.hidden_nodes[1], 1) # 4
#         Connection(None, self.phenotype.hidden_nodes[1], self.phenotype.output_nodes[1], 1) # 5
#         Connection(None, self.phenotype.input_nodes[2], self.phenotype.hidden_nodes[2], 1) # 6
#         Connection(None, self.phenotype.hidden_nodes[2], self.phenotype.hidden_nodes[1], 1) # 7
#         Connection(None, self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[2], 1) # 8

#         sorted_nodes = self.phenotype.get_topsorted_nodes(self.phenotype.all_nodes)

#         node_list = [
#             self.phenotype.input_nodes[2],
#             self.phenotype.hidden_nodes[2],
#             self.phenotype.output_nodes[2],
#             self.phenotype.input_nodes[1],
#             self.phenotype.input_nodes[0],
#             self.phenotype.hidden_nodes[0],
#             self.phenotype.hidden_nodes[1],
#             self.phenotype.output_nodes[1],
#             self.phenotype.output_nodes[0]
#         ]

#         self.assertEquals(sorted_nodes, node_list)

#     def test_get_topsorted_nodes2(self):
#         self.phenotype._input_nodes = [Node(None)]
#         self.phenotype._hidden_nodes = [Node(None) for i in range(3)]
#         self.phenotype._output_nodes = [Node(None)]

#         Connection(None, self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[1], 1) # 0
#         Connection(None, self.phenotype.hidden_nodes[1], self.phenotype.hidden_nodes[0], 1) # 1
#         Connection(None, self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[2], 1) # 2
#         Connection(None, self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[0], 1) # 3

#         sorted_nodes = self.phenotype.get_topsorted_nodes(self.phenotype.all_nodes)

#         node_list = [
#             self.phenotype.input_nodes[0],
#             self.phenotype.hidden_nodes[1],
#             self.phenotype.hidden_nodes[0],
#             self.phenotype.hidden_nodes[2],
#             self.phenotype.output_nodes[0]
#         ]

#         self.assertEquals(sorted_nodes, node_list)

#     def test_update_locations1(self):
#         self.phenotype._input_nodes = [Node(None)]
#         self.phenotype._hidden_nodes = [Node(None) for i in range(3)]
#         self.phenotype._output_nodes = [Node(None)]

#         Connection(None, self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[1], 1) # 0
#         Connection(None, self.phenotype.hidden_nodes[1], self.phenotype.hidden_nodes[0], 1) # 1
#         Connection(None, self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[2], 1) # 2
#         Connection(None, self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[0], 1) # 3

#         self.phenotype.update_locations()
        
#         locations = [node.location for node in self.phenotype.all_nodes]

#         self.assertEquals(locations, [0, 2, 1, 3, 4])

#     def test_update_locations2(self):
#         self.phenotype._input_nodes = [Node(None) for i in range(3)]
#         self.phenotype._hidden_nodes = [Node(None) for i in range(3)]
#         self.phenotype._output_nodes = [Node(None) for i in range(3)]

#         Connection(None, self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[0], 1) # 0
#         Connection(None, self.phenotype.hidden_nodes[0], self.phenotype.output_nodes[0], 1) # 1
#         Connection(None, self.phenotype.input_nodes[0], self.phenotype.output_nodes[1], 1) # 2
#         Connection(None, self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[1], 1) # 3
#         Connection(None, self.phenotype.input_nodes[1], self.phenotype.hidden_nodes[1], 1) # 4
#         Connection(None, self.phenotype.hidden_nodes[1], self.phenotype.output_nodes[1], 1) # 5
#         Connection(None, self.phenotype.input_nodes[2], self.phenotype.hidden_nodes[2], 1) # 6
#         Connection(None, self.phenotype.hidden_nodes[2], self.phenotype.hidden_nodes[1], 1) # 7
#         Connection(None, self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[2], 1) # 8

#         self.phenotype.update_locations()
        
#         locations = [node.location for node in self.phenotype.all_nodes]

#         self.assertEquals(locations, [0, 0, 0, 1, 2, 1, 2, 3, 2])

#     def test_update_locations3(self):
#         self.phenotype._input_nodes = [Node(None, i + 1) for i in range(3)]
#         self.phenotype._hidden_nodes = [Node(None, i + 5) for i in range(5)]
#         self.phenotype._output_nodes = [Node(None, 4)]

#         conn0 = Connection(None, self.phenotype.input_nodes[0], self.phenotype.hidden_nodes[0], 1) # 0
#         conn1 = Connection(None, self.phenotype.hidden_nodes[0], self.phenotype.hidden_nodes[1], 1) # 1
#         conn2 = Connection(None, self.phenotype.hidden_nodes[1], self.phenotype.hidden_nodes[2], 1) # 2
#         conn3 = Connection(None, self.phenotype.hidden_nodes[2], self.phenotype.output_nodes[0], 1) # 3
#         conn4 = Connection(None, self.phenotype.input_nodes[1], self.phenotype.hidden_nodes[3], 1) # 4
#         conn5 = Connection(None, self.phenotype.hidden_nodes[3], self.phenotype.output_nodes[0], 1) # 5
#         conn6 = Connection(None, self.phenotype.input_nodes[2], self.phenotype.hidden_nodes[4], 1) # 6
#         conn7 = Connection(None, self.phenotype.hidden_nodes[4], self.phenotype.output_nodes[0], 1) # 7
#         conn8 = Connection(None, self.phenotype.hidden_nodes[2], self.phenotype.hidden_nodes[3], 1) # 8
#         conn9 = Connection(None, self.phenotype.hidden_nodes[4], self.phenotype.hidden_nodes[4], 1) # 9

#         self.phenotype.hidden_nodes[3].rcc = True
#         conn8.recurrent = True
#         self.phenotype.hidden_nodes[4].rcc = True
#         conn9.recurrent = True

#         self.phenotype.update_locations()
        
#         locations = [node.location for node in self.phenotype.all_nodes]

#         self.assertEquals(locations, [0, 0, 0, 1, 2, 3, 1, 1, 4])
