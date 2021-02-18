import unittest

from kypy_neat.phenotype import Phenotype, StabilizationMethod
from kypy_neat.genotype import Genotype
from kypy_neat.traits import Node, Connection
from kypy_neat.genes import NodeType, gene_factory
from kypy_neat.utils.math import sigmoid


class TestPhenotype(unittest.TestCase):
    def setUp(self):
        gene_factory.reset()

    def build_non_recurrent_nn1(self):
        genotype = Genotype()
        genotype.create_node_gene(None, None, NodeType.INPUT) # 1
        genotype.create_node_gene(None, None, NodeType.OUTPUT) # 2

        genotype.create_connection_gene(1, 2, 1)

        phenotype = Phenotype(genotype)
        
        return phenotype

    def build_non_recurrent_nn2(self):
        genotype = Genotype()
        genotype.create_node_gene(None, None, NodeType.INPUT) # 1
        genotype.create_node_gene(None, None, NodeType.OUTPUT) # 2

        genotype.create_node_gene(1, 2, NodeType.HIDDEN) # 3

        genotype.create_connection_gene(1, 3, 1)
        genotype.create_connection_gene(3, 2, 1)

        phenotype = Phenotype(genotype)
        
        return phenotype

    def build_non_recurrent_nn3(self):
        genotype = Genotype()
        genotype.create_node_gene(None, None, NodeType.INPUT) # 1
        genotype.create_node_gene(None, None, NodeType.INPUT) # 2
        genotype.create_node_gene(None, None, NodeType.INPUT) # 3
        genotype.create_node_gene(None, None, NodeType.OUTPUT) # 4

        genotype.create_node_gene(1, 4, NodeType.HIDDEN) # 5
        genotype.create_node_gene(2, 4, NodeType.HIDDEN) # 6
        genotype.create_node_gene(3, 4, NodeType.HIDDEN) # 7

        genotype.create_connection_gene(1, 5, 1)
        genotype.create_connection_gene(5, 4, 1)

        genotype.create_connection_gene(2, 6, 1)
        genotype.create_connection_gene(6, 4, 1)

        genotype.create_connection_gene(3, 7, 1)
        genotype.create_connection_gene(7, 4, 1)

        genotype.create_connection_gene(7, 6, 1)
        phenotype = Phenotype(genotype)

        return phenotype

    def build_non_recurrent_nn4(self):
        genotype = Genotype()
        genotype.create_node_gene(None, None, NodeType.INPUT) # 1
        genotype.create_node_gene(None, None, NodeType.INPUT) # 2
        genotype.create_node_gene(None, None, NodeType.INPUT) # 3
        genotype.create_node_gene(None, None, NodeType.OUTPUT) # 4

        genotype.create_node_gene(1, 4, NodeType.HIDDEN) # 5
        genotype.create_node_gene(2, 4, NodeType.HIDDEN) # 6
        genotype.create_node_gene(3, 4, NodeType.HIDDEN) # 7

        genotype.create_connection_gene(1, 5, 1)
        genotype.create_connection_gene(5, 4, 1)

        genotype.create_connection_gene(2, 6, 1)
        genotype.create_connection_gene(6, 4, 1)
        
        genotype.create_connection_gene(6, 6, 1)

        genotype.create_connection_gene(3, 7, 1)
        genotype.create_connection_gene(7, 4, 1)

        genotype.create_connection_gene(7, 6, 1)
        phenotype = Phenotype(genotype)

        return phenotype

    def build_recurrent_nn1(self):
        genotype = Genotype()
        genotype.create_node_gene(None, None, NodeType.INPUT) # 1
        genotype.create_node_gene(None, None, NodeType.INPUT) # 2
        genotype.create_node_gene(None, None, NodeType.INPUT) # 3
        genotype.create_node_gene(None, None, NodeType.OUTPUT) # 4

        genotype.create_node_gene(1, 6, NodeType.HIDDEN) # 5
        genotype.create_node_gene(5, 4, NodeType.HIDDEN) # 6

        genotype.create_connection_gene(1, 4, 1)
        genotype.create_connection_gene(1, 5, 1)
        genotype.create_connection_gene(2, 5, 1)
        genotype.create_connection_gene(3, 4, 1)
        genotype.create_connection_gene(5, 6, 1)
        genotype.create_connection_gene(6, 4, 1)
        genotype.create_connection_gene(6, 5, 1)

        phenotype = Phenotype(genotype)

        return phenotype

    def test_activate_1_pass(self):
        phenotype = self.build_recurrent_nn1()
        expected = sigmoid(1)
        phenotype.activate([1, 1, 0])
        output = phenotype.output_nodes[0].activation
        self.assertEquals(output, expected)
        
    def test_activate_2_pass(self):
        phenotype = self.build_recurrent_nn1()
        expected = sigmoid(1.5)
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        output = phenotype.output_nodes[0].activation
        self.assertEquals(output, expected)

    def test_activate_3_pass(self):
        phenotype = self.build_recurrent_nn1()
        expected = sigmoid(sigmoid(sigmoid(2)) + 1)
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        output = phenotype.output_nodes[0].activation
        self.assertEquals(output, expected)

    def test_activate_4_pass(self):
        phenotype = self.build_recurrent_nn1()
        expected = sigmoid(sigmoid(sigmoid(2.5)) + 1)
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        output = phenotype.output_nodes[0].activation
        self.assertEquals(output, expected)

    def test_activate_5_pass(self):
        phenotype = self.build_recurrent_nn1()
        expected = sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(2)) + 2)) + 1)
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        phenotype.activate([1, 1, 0])
        output = phenotype.output_nodes[0].activation
        self.assertEquals(output, expected)

    def test_evaluate_network_iterative_5_pass(self):
        phenotype = self.build_recurrent_nn1()
        phenotype._stabilization_method = StabilizationMethod.ITERATIVE
        phenotype._iteration_limit = 5
        expected = [sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(2)) + 2)) + 1)]
        output, *_ = phenotype.evaluate_network([1, 1, 0])
        self.assertEquals(output, expected)

    def test_evaluate_network_iterative_100_pass(self):
        phenotype = self.build_recurrent_nn1()
        phenotype._stabilization_method = StabilizationMethod.ITERATIVE
        phenotype._iteration_limit = 100
        expected = [sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(2)) + 2)) + 1)]
        output, *_ = phenotype.evaluate_network([1, 1, 0])

        self.assertAlmostEquals(output[0], expected[0], delta=0.0001)

    def test_evaluate_network_output_delta(self):
        phenotype = self.build_recurrent_nn1()
        phenotype._stabilization_method = StabilizationMethod.OUTPUT_DELTA
        expected = [sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(2)) + 2)) + 1)]
        output, *_ = phenotype.evaluate_network([1, 1, 0])
        self.assertAlmostEquals(output[0], expected[0], delta=0.0001)

    def test_multiple_network_evaluations(self):
        phenotype = self.build_recurrent_nn1()
        phenotype._stabilization_method = StabilizationMethod.ITERATIVE
        phenotype._iteration_limit = 5
        expected = [sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(2)) + 2)) + 1)]
        for _ in range(100):
            output, *_ = phenotype.evaluate_network([1, 1, 0])
        self.assertEquals(output, expected)

    def test_stabilization_comparison(self):
        phenotype = self.build_recurrent_nn1()
        for node in phenotype.all_nodes:
            node._stability_threshold = 1E-16
        phenotype._stabilization_method = StabilizationMethod.ITERATIVE
        phenotype._iteration_limit = 100
        output1, *_ = phenotype.evaluate_network([1, 1, 0])
        phenotype._stabilization_method = StabilizationMethod.OUTPUT_DELTA
        output2, *_ = phenotype.evaluate_network([1, 1, 0])
        self.assertAlmostEquals(output1, output2, delta=1E-16)