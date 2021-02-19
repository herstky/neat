from enum import Enum

class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Gene:
    def __init__(self, innovation_id, genotype, input_node_id, output_node_id):
        self._innovation_id = innovation_id
        self._genotype = genotype
        self._input_node_id = input_node_id
        self._output_node_id = output_node_id

    @property
    def innovation_id(self):
        return self._innovation_id

    @property
    def genotype(self):
        return self._genotype

    @property
    def input_node_id(self):
        return self._input_node_id

    @property
    def output_node_id(self):
        return self._output_node_id

    @property
    def structure(self):
        return (self._input_node_id, self._output_node_id)


class NodeGene(Gene):
    def __init__(self, innovation_id, genotype, input_node_id, output_node_id, node_type):
        super().__init__(innovation_id, genotype, input_node_id, output_node_id)
        self._node_type = node_type

    @property
    def node_type(self):
        return self._node_type

class ConnectionGene(Gene):
    def __init__(self, innovation_id, genotype, input_node_id, output_node_id, weight, enabled=True):
        super().__init__(innovation_id, genotype, input_node_id, output_node_id)
        self.weight = weight
        self.enabled = enabled

class _GeneFactory:
    def __init__(self):
        self.reset()

    def reset(self):
        # Containers are initialized with a single dummy value to shift array
        # indices to match ids since ids start at 1
        self._node_list = [None]
        self._node_dict = {None: 0}
        self._connection_list = [None]
        self._connection_dict = {None: 0}

    def get_num_nodes(self):
        return len(self._node_list) - 1

    def get_num_connections(self):
        return len(self._connection_list) - 1

    def get_nodes(self):
        return self._node_list[1:]

    def get_connections(self):
        return self._connection_list[1:]

    def get_node_id(self, structure):
        if structure in self._node_dict:
            return self._node_dict[structure]
        else:
            return None

    def get_connection_id(self, structure):
        if structure in self._connection_dict:
            return self._connection_dict[structure]
        else:
            return None

    def get_node_structure(self, innovation_id):
        if innovation_id <= self.get_num_nodes:
            return self._node_list[innovation_id]
        else:
            return None

    def get_connection_structure(self, innovation_id):
        if innovation_id <= self.get_num_connections:
            return self._connection_list[innovation_id]
        else:
            return None
    
    def create_node_gene(self, genotype, input_node_id, output_node_id, node_type):
        structure = (input_node_id, output_node_id)
        if self.node_innovation_exists(structure) and node_type not in (NodeType.INPUT, NodeType.OUTPUT):
            innovation_id = self.get_node_id(structure)
        else:
            self._node_list.append(node_type)
            innovation_id = self.get_num_nodes()
            self._node_dict[structure] = innovation_id

        return NodeGene(innovation_id, genotype, input_node_id, output_node_id, node_type)

    def copy_node_gene(self, node_gene):
        return NodeGene(node_gene.innovation_id, 
                        node_gene.genotype,
                        node_gene.input_node_id, 
                        node_gene.output_node_id, 
                        node_gene.node_type)

    def create_connection_gene(self, genotype, input_node_id, output_node_id, weight, enabled=True):
        structure = (input_node_id, output_node_id)
        if self.connection_innovation_exists(structure):
            innovation_id = self.get_connection_id(structure)
        else:
            self._connection_list.append(structure)
            innovation_id = self.get_num_connections()
            self._connection_dict[structure] = innovation_id

        return ConnectionGene(innovation_id, genotype, input_node_id, output_node_id, weight, enabled)

    def copy_connection_gene(self, connection_gene):
        return ConnectionGene(connection_gene.innovation_id,
                              connection_gene.genotype,
                              connection_gene.input_node_id,
                              connection_gene.output_node_id,
                              connection_gene.weight,
                              connection_gene.enabled)

    def node_innovation_exists(self, structure):
        return structure in self._node_dict

    def connection_innovation_exists(self, structure):
        return structure in self._connection_dict


gene_factory = _GeneFactory()
