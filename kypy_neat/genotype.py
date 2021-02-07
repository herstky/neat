from kypy_neat.genes import genetic_history

class Genotype:
    def __init__(self):
        self._node_genes = []
        self._connection_genes = []

    @property
    def node_genes(self):
        return self._node_genes

    @property
    def connection_genes(self):
        return self._connection_genes

    def create_node_gene(self, node_type):
        node_gene = genetic_history.create_node_gene(node_type)
        self._node_genes.append(node_gene)
        return node_gene

    def create_connection_gene(self, input_node_id, output_node_id, weight, enabled=True, recurrent=False):
        connection_gene = genetic_history.create_connection_gene(input_node_id, output_node_id, weight, enabled, recurrent)
        self._connection_genes.append(connection_gene)
        return connection_gene