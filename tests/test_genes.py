import unittest

from kypy_neat.genes import NodeType, NodeGene, ConnectionGene, genetic_history

class TestNodeGene(unittest.TestCase):
    def setUp(self):
        self.node_gene = NodeGene(1, None, NodeType.INPUT)

    def test_id(self):
        self.assertEquals(self.node_gene.node_id, 1)

    def test_node_type(self):
        self.assertEquals(self.node_gene.node_type, NodeType.INPUT)

    def test_rcc(self):
        self.assertEquals(self.node_gene.rcc, False)

class TestConnectionGene(unittest.TestCase):
    def setUp(self):
        self.connection_gene = ConnectionGene(1, None, 1, 2, 1)

    def test_innovation_id(self):
        self.assertEquals(self.connection_gene.innovation_id, 1)

    def test_input_node_id(self):
        self.assertEquals(self.connection_gene.input_node_id, 1)

    def test_output_node_id(self):
        self.assertEquals(self.connection_gene.output_node_id, 2)

    def test_structure1(self):
        self.assertEquals(self.connection_gene.structure, (1, 2))

    def test_structure2(self):
        self.assertNotEquals(self.connection_gene.structure, (2, 1))

    def test_structure3(self):
        self.assertNotEquals(self.connection_gene.structure, tuple())

    def test_structure4(self):
        self.assertNotEquals(self.connection_gene.structure, (3, 4))

class TestGeneticHistory(unittest.TestCase):
    def setUp(self):
        genetic_history.reset()
        genetic_history.create_node_gene(None, NodeType.INPUT) # 1
        genetic_history.create_node_gene(None, NodeType.INPUT) # 2
        genetic_history.create_node_gene(None, NodeType.OUTPUT) # 3
        genetic_history.create_node_gene(None, NodeType.HIDDEN) # 4
        genetic_history.create_node_gene(None, NodeType.HIDDEN) # 5
        genetic_history.create_node_gene(None, NodeType.HIDDEN) # 6

        genetic_history.create_connection_gene(None, 1, 3, 1) # 1
        genetic_history.create_connection_gene(None, 1, 4, 1) # 2
        genetic_history.create_connection_gene(None, 1, 6, 1) # 3
        genetic_history.create_connection_gene(None, 4, 5, 1) # 4
        genetic_history.create_connection_gene(None, 5, 3, 1) # 5
        genetic_history.create_connection_gene(None, 2, 6, 1) # 6
        genetic_history.create_connection_gene(None, 6, 3, 1) # 7
    
    def test_create_node_gene(self):
        self.assertEquals(genetic_history._node_list[1], NodeType.INPUT)


    def test_create_connection_gene1(self):
        self.assertEquals(genetic_history._connection_list[1], (1, 3))

    def test_create_connection_gene2(self):
        connection_gene = genetic_history.create_connection_gene(None, 1, 6, 1)
        self.assertEquals(connection_gene.innovation_id, 3)
    
    def test_create_connection_gene3(self):
        connection_gene = genetic_history.create_connection_gene(None, 2, 5, 1)
        self.assertEquals(connection_gene.innovation_id, 8)

    def test_get_num_nodes(self):
        self.assertEquals(genetic_history.get_num_nodes(), 6)

    def test_get_num_connections(self):
        self.assertEquals(genetic_history.get_num_connections(), 7)

    def test_get_connections(self):
        self.assertEquals(genetic_history.get_connections(), [(1, 3), (1, 4), (1, 6), (4, 5), (5, 3), (2, 6), (6, 3)])

    def test_get_innovation_id(self):
        structure = (1, 6)
        self.assertEquals(genetic_history.get_innovation_id(structure), 3)

    def test_structure_exists1(self):
        structure = (1, 3)
        self.assertTrue(genetic_history.structure_exists(structure))

    def test_structure_exists2(self):
        structure = (4, 1)
        self.assertFalse(genetic_history.structure_exists(structure))

    def test_structure_exists3(self):
        structure = (2, 5)
        self.assertFalse(genetic_history.structure_exists(structure))

    def test_connection_gene_exists1(self):
        connection_gene = genetic_history.create_connection_gene(None, 1, 6, 1)
        self.assertTrue(genetic_history.connection_gene_exists(connection_gene))

    def test_connection_gene_exists2(self):
        connection_gene = genetic_history.create_connection_gene(None, 1, 3, 1)
        self.assertTrue(genetic_history.connection_gene_exists(connection_gene))

    def test_connection_gene_exists3(self):
        connection_gene = ConnectionGene(1, None, 2, 5, 1)
        self.assertFalse(genetic_history.connection_gene_exists(connection_gene))





