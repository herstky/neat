from kypy_neat.genes import NodeGene, ConnectionGene
from kypy_neat.traits import Node, Connection

class Phenotype:
    def __init__(self, genotype):
        self._genotype = genotype
        self._input_nodes = []
        self._hidden_nodes = []
        self._output_nodes = []
        self._node_map = {}
        self._connection_map = {}

    @property
    def genotype(self):
        return self._genotype

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def hidden_nodes(self):
        return self._hidden_nodes

    @property
    def output_nodes(self):
        return self._output_nodes

    @property
    def inner_nodes(self):
        return self._input_nodes + self._hidden_nodes

    @property
    def outer_nodes(self):
        return self._hidden_nodes + self._output_nodes

    @property
    def all_nodes(self):
        return self._input_nodes + self._hidden_nodes + self._output_nodes        

    def _build(self):
        for node_gene in self._genotype.node_genes:
            node = self.generate_node(node_gene)

        for connection_gene in self._genotype.connection_genes:
            connection = self.generate_connection(connection_gene)

    def generate_node(self, gene):
        node = Node(gene)
        self._node_map[gene.node_id] = node
        if gene.Type is NodeType.INPUT:
            self._input_nodes.append(node)
        elif gene.Type is NodeType.HIDDEN:
            self._hidden_nodes.append(node)
        elif gene.Type is NodeType.OUTPUT:
            self._output_nodes.append(node)

        return node

    def generate_connection(self, gene):
        input_node = self._node_map[gene.input_node_id]
        output_node = self._node_map[gene._output_node_id]
        connection = Connection(gene, input_node, output_node)
        self._connection_map[gene.innovation_id] = connection

        return connection

    def _topsort_dfs(self, ordering_idx, ordering, visited, node):
        visited.add(node)

        for conn in node.output_connections:
            if not conn.recurrent and conn.output_node not in visited:
                ordering_idx = self._topsort_dfs(ordering_idx, ordering, visited, conn.output_node)

        ordering[ordering_idx] = node
        return ordering_idx - 1

    def get_topsorted_nodes(self, nodes):
        num_nodes = len(nodes)
        visited = set()
        ordering = [None] * num_nodes
        ordering_idx = num_nodes - 1

        for node in nodes:
            if node not in visited:
                ordering_idx = self._topsort_dfs(ordering_idx, ordering, visited, node)

        return ordering

    def _flip_location_signs(self, nodes):
        for node in nodes:
            node.location *= -1

    def _single_source_longest_path(self, nodes):
        sorted_nodes = self.get_topsorted_nodes(nodes)
        self._flip_location_signs(sorted_nodes)
        for node in sorted_nodes:
            for conn in node.output_connections:
                if conn.recurrent:
                    continue
                neighbor = conn.output_node
                neighbor.location = min(neighbor.location, node.location - 1)

        self._flip_location_signs(sorted_nodes)

    def update_locations(self):
        for node in self.all_nodes:
            node.location = float('-inf')

        for node in self._input_nodes:
            node.location = 0
            nodes = [node] + self.outer_nodes
            self._single_source_longest_path(nodes)

 




