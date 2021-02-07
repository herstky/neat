# have a map of node_id to node and connection_id to connection
# when crossing over genotypes, first add all of the nodes (including 
# disjoint/excess nodes) of the fittest parent to the offspring, 
# then add all connections of the fittest parent

class NEAT:
    class History:
        def __init__(self):
            self._node_list = []
            self._conn_list = []
            self._conn_set = []

        def add_node(self, node):
            self._node_list.append(node)
            return len(self._node_list) - 1

        def add_connection(self, input_node, output_node):
            self._conn_list.append()
            pass

    def __init__(self):
        self.node_map = {}
        self.connection_map = {}
        self.innovation_history = []

    def _compatibility_function(self):
        pass

    
    