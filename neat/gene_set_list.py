from copy import deepcopy

class GeneSetList:
    def __init__(self):
        self._genes = []
        self._structures = set()

    def add_gene(self, gene):
        index = self._get_index_of_new_gene(gene)
        self._insert_gene(index, gene)

    def _insert_gene(self, index, gene):
        self._genes.insert(index, gene)
        self._structures.add(gene.structure)

    def _get_index_of_new_gene(self, gene):
        index = 0
        for existing_gene in self._genes:
            if gene.innovation_id > existing_gene.innovation_id:
                index += 1
        return index

    @property
    def genes(self):
        return self._genes

    @property
    def structures(self):
        return self.structures

    def contains_structure(self, structure):
        return structure in self._structures

    def generate_copy(self):
        return deepcopy(self)