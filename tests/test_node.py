import unittest
import math

from kypy_neat.utils.math import sigmoid
from kypy_neat.traits import Node


class TestNode(unittest.TestCase):
    def setUp(self):
        self.node = Node(None)

    def test_activation_function1(self):
        self.assertEquals(self.node.activation_function(0), 0.5)

    def test_activation_function2(self):
        self.assertEquals(self.node.activation_function(1), sigmoid(1))

    def test_activation_function3(self):
        self.assertEquals(self.node.activation_function(-1), sigmoid(-1))


