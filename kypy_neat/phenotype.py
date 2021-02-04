class Phenotype:
    def __init__(self):
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []

    @property
    def all_nodes(self):
        return self.input_nodes + self.hidden_nodes + self.output_nodes

    def _dfs_helper(self, ordering_idx, ordering, visited, node):
        visited.add(node)

        for conn in node.output_connections:
            if not conn.recurrent and conn.output_node not in visited:
                ordering_idx = self._dfs_helper(ordering_idx, ordering, visited, conn.output_node)

        ordering[ordering_idx] = node
        return ordering_idx - 1

    def topsorted_nodes(self, nodes):
        num_nodes = len(nodes)
        visited = set()
        ordering = [None] * num_nodes
        ordering_idx = num_nodes - 1

        for node in nodes:
            if node not in visited:
                ordering_idx = self._dfs_helper(ordering_idx, ordering, visited, node)

        return ordering

    def _flip_location_signs(self, nodes):
        for node in nodes:
            node.location *= -1

    def _single_source_longest_path(self, nodes):
        sorted_nodes = self.topsorted_nodes(nodes)
        self._flip_location_signs(sorted_nodes)
        for node in sorted_nodes:
            for conn in node.output_connections:
                neighbor = conn.output_node
                neighbor.location = min(neighbor.location, node.location - 1)

        locations = [node.location for node in sorted_nodes]
        self._flip_location_signs(sorted_nodes)

    def update_locations(self):
        for node in self.all_nodes:
            node.location = float('-inf')

        for node in self.input_nodes:
            node.location = 0
            nodes = [node] + self.hidden_nodes + self.output_nodes
            self._single_source_longest_path(nodes)




