from enum import Enum

class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Gene:
    def __init__(self, id_, genotype):
        self._genotype = genotype
        self._id = id_

    @property
    def genotype(self):
        return self._genotype

class NodeGene(Gene):
    def __init__(self, id_, genotype, node_type):
        super().__init__(id_, genotype)
        self._node_type = node_type

    @property
    def node_id(self):
        return self._id

    @property
    def node_type(self):
        return self._node_type

class ConnectionGene(Gene):
    def __init__(self, id_, genotype, input_node_id, output_node_id, weight, enabled=True):
        super().__init__(id_, genotype)
        self._input_node_id = input_node_id
        self._output_node_id = output_node_id
        self.weight = weight
        self.enabled = enabled

    @property
    def innovation_id(self):
        return self._id

    @property
    def input_node_id(self):
        return self._input_node_id

    @property
    def output_node_id(self):
        return self._output_node_id

    @property
    def structure(self):
        return (self._input_node_id, self._output_node_id)


class _GeneticHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        # Containers are initialized with a single dummy value to shift array
        # indices to match ids since ids start at 1
        self._node_list = [None]
        self._connection_list = [None]
        self._connection_dict = {None: 0}

    def get_num_nodes(self):
        return len(self._node_list) - 1

    def get_num_connections(self):
        return len(self._connection_list) - 1

    def get_connections(self):
        return self._connection_list[1:]

    def get_innovation_id(self, structure):
        if structure in self._connection_dict:
            return self._connection_dict[structure]
        else:
            return None

    def get_structure(self, innovation_id):
        if innovation_id <= self.get_num_connections:
            return self._connection_list[innovation_id]
        else:
            return None
        
    def create_node_gene(self, genotype, node_type):
        self._node_list.append(node_type)
        node_id = self.get_num_nodes()
        return NodeGene(node_id, genotype, node_type)

    def create_connection_gene(self, genotype, input_node_id, output_node_id, weight, enabled=True):
        structure = (input_node_id, output_node_id)
        if self.structure_exists(structure):
            innovation_id = self.get_innovation_id(structure)
        else:
            self._connection_list.append(structure)
            innovation_id = self.get_num_connections()
            self._connection_dict[structure] = innovation_id

        return ConnectionGene(innovation_id, genotype, input_node_id, output_node_id, weight, enabled)

    def structure_exists(self, structure):
        return structure in self._connection_dict

    def connection_gene_exists(self, connection_gene):
        structure = (connection_gene.input_node_id, connection_gene.output_node_id)
        return self.structure_exists(structure)


genetic_history = _GeneticHistory()
