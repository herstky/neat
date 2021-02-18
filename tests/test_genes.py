import unittest

from kypy_neat.genes import NodeType, NodeGene, ConnectionGene, gene_factory

class TestNodeGene(unittest.TestCase):
    def setUp(self):
        self.node_gene = NodeGene(1, None, None, None, NodeType.INPUT)

    def test_id(self):
        self.assertEquals(self.node_gene.innovation_id, 1)

    def test_node_type(self):
        self.assertEquals(self.node_gene.node_type, NodeType.INPUT)

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

class TestGeneFactory(unittest.TestCase):
    def setUp(self):
        gene_factory.reset()
        gene_factory.create_node_gene(None, None, None, NodeType.INPUT) # Node 1
        gene_factory.create_node_gene(None, None, None, NodeType.INPUT) # Node 2
        gene_factory.create_node_gene(None, None, None, NodeType.OUTPUT) # Node 3

        gene_factory.create_connection_gene(None, 1, 3, 1) # Connection 1
        gene_factory.create_connection_gene(None, 2, 3, 1) # Connection 2

        gene_factory.create_node_gene(None, 1, 3, NodeType.HIDDEN) # Node 4
        gene_factory.create_node_gene(None, 2, 3, NodeType.HIDDEN) # Node 5

        gene_factory.create_connection_gene(None, 1, 4, 1) # Connection 3
        gene_factory.create_connection_gene(None, 4, 3, 1) # Connection 4
        gene_factory.create_connection_gene(None, 2, 5, 1) # Connection 5
        gene_factory.create_connection_gene(None, 5, 3, 1) # Connection 6
        
        gene_factory.create_connection_gene(None, 1, 5, 1) # Connection 7

    def test_create_node_gene(self):
        self.assertEquals(gene_factory._node_list[1], NodeType.INPUT)


    def test_create_connection_gene1(self):
        self.assertEquals(gene_factory._connection_list[1], (1, 3))

    def test_create_connection_gene2(self):
        connection_gene = gene_factory.create_connection_gene(None, 1, 5, 1)
        self.assertEquals(connection_gene.innovation_id, 7)
    
    def test_create_connection_gene3(self):
        connection_gene = gene_factory.create_connection_gene(None, 2, 4, 1)
        self.assertEquals(connection_gene.innovation_id, 8)

    def test_create_connection_gene4(self):
        connection_gene = gene_factory.create_connection_gene(None, 1, 4, 1)
        self.assertEquals(connection_gene.innovation_id, 3)

    def test_get_num_nodes(self):
        self.assertEquals(gene_factory.get_num_nodes(), 5)

    def test_get_num_connections(self):
        self.assertEquals(gene_factory.get_num_connections(), 7)

    def test_get_connections(self):
        self.assertEquals(gene_factory.get_connections(), [(1, 3), (2, 3), (1, 4), (4, 3), (2, 5), (5, 3), (1, 5)])

    def test_get_innovation_id(self):
        structure = (4, 3)
        self.assertEquals(gene_factory.get_connection_id(structure), 4)

    def test_connection_exists1(self):
        structure = (1, 3)
        self.assertTrue(gene_factory.connection_innovation_exists(structure))

    def test_connection_exists2(self):
        structure = (4, 1)
        self.assertFalse(gene_factory.connection_innovation_exists(structure))

    def test_connection_exists3(self):
        structure = (2, 6)
        self.assertFalse(gene_factory.connection_innovation_exists(structure))






