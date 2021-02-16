import random as rand
import time

from kypy_neat.genes import NodeType, ConnectionGene, NodeGene, genetic_history
from kypy_neat.utils.timer import timer


class Genotype:
    base_genotype = None
    _mutate_starting_topologies = False

    _weight_power = 8
    _weight_cap = 8.0

    _weight_mutation_chance = 0.7 # chance of a weight being perturbed
    _weight_cold_mutation_chance = 0 # chance of a weight being replaced by a random weight
    _end_genotype_threshold = 0.80 # point after which weight mutations are less likely
    _end_weight_mutation_change = 0.7
    _end_weight_cold_mutation_chance = 0
    
    _node_mutation_chance = 0.05
    _connection_mutation_chance = 0.08

    _toggle_chance = 0.0
    _reenable_chance = 0.0

    _excess_coeff = 1
    _disjoint_coeff = 1
    _weight_coeff = 0.4
    

    def __init__(self):
        self._node_genes = []
        self._connection_genes = []
        self._connection_structures = set()
        self._innovation_count = 0

    def generate_copy(self):
        genotype_copy = Genotype()
        for node_gene in self._node_genes:
            node_gene_copy = NodeGene(node_gene.node_id, genotype_copy, node_gene.node_type)
            genotype_copy._node_genes.append(node_gene_copy)

        for connection_gene in self._connection_genes:
            connection_gene_copy = ConnectionGene(connection_gene.innovation_id, 
                                                  genotype_copy, 
                                                  connection_gene.input_node_id, 
                                                  connection_gene.output_node_id, 
                                                  connection_gene.weight, 
                                                  connection_gene.enabled)
            genotype_copy._connection_genes.append(connection_gene_copy)
            genotype_copy._connection_structures.add(connection_gene_copy.structure)
 

        return genotype_copy

    @property
    def innovation_count(self):
        return self._innovation_count

    @classmethod
    def initialize_minimal_topology(cls, num_inputs, num_outputs):
        cls.base_genotype = Genotype()
        input_nodes = []
        output_nodes = []
        for _ in range(num_inputs):
            input_nodes.append(cls.base_genotype.create_node_gene(NodeType.INPUT))

        for _ in range(num_outputs):
            output_nodes.append(cls.base_genotype.create_node_gene(NodeType.OUTPUT))


        for i in range(len(input_nodes)):
            for j in range(len(output_nodes)):
                input_node = input_nodes[i]
                output_node = output_nodes[j]
                cls.base_genotype.create_connection_gene(input_node.node_id, output_node.node_id, cls.generate_weight_modifier())

    @classmethod
    def base_genotype_factory(cls):
        new_genotype = cls.base_genotype.generate_copy()
        if cls._mutate_starting_topologies:
            new_genotype.mutate_weights()
            new_genotype.attempt_topological_mutations()

        return new_genotype

    @property
    def node_genes(self):
        return self._node_genes

    @property
    def connection_genes(self):
        return self._connection_genes

    def structure_exists(self, structure):
        return structure in self._connection_structures

    def mutate_weights(self):
        for i, conn in enumerate(self._connection_genes):
            if i / len(self._connection_genes) < self._end_genotype_threshold:
                mutation_chance = self._end_weight_mutation_change
                cold_mutation_chance = self._end_weight_cold_mutation_chance
     
            else:
                mutation_chance = self._weight_mutation_chance
                cold_mutation_chance = self._weight_cold_mutation_chance

            weight_mod = self.generate_weight_modifier()
            if rand.uniform(0, 1) < mutation_chance:
                conn.weight += weight_mod
            elif rand.uniform(0, 1) < cold_mutation_chance:
                conn.weight = weight_mod
            
            conn.weight = min(max(conn.weight, -self._weight_cap), self._weight_cap)

    def mutate(self):
        self.mutate_weights()
        self.mutate_connection_states()

    @classmethod
    def generate_weight_modifier(cls):
        return rand.uniform(-1, 1) * cls._weight_power

    def create_node_gene(self, node_type):
        node_gene = genetic_history.create_node_gene(self, node_type)
        self._node_genes.append(node_gene)
        return node_gene

    def create_connection_gene(self, input_node_id, output_node_id, weight, enabled=True):
        connection_gene = genetic_history.create_connection_gene(self, input_node_id, output_node_id, weight, enabled)
        self._connection_genes.append(connection_gene)
        self._connection_structures.add(connection_gene.structure)
        return connection_gene

    def attempt_connection_mutation(self):
        input_node = self._node_genes[rand.randint(0, len(self._node_genes) - 1)]  
        output_node = self._node_genes[rand.randint(0, len(self._node_genes) - 1)]
        structure = (input_node.node_id, output_node.node_id)
        weight = self.generate_weight_modifier()
        if not self.structure_exists(structure):
            self.create_connection_gene(input_node.node_id, output_node.node_id, weight)
            # print(f'Connection mutated between nodes {input_node.node_id} and {output_node.node_id}')
            return True

        return False

    def attempt_node_mutation(self):
        if not len(self._connection_genes):
            return False

        conn_to_split = self._connection_genes[rand.randint(0, len(self._connection_genes) - 1)]
        conn_to_split.enabled = False
        new_node = self.create_node_gene(NodeType.HIDDEN)
        new_input_conn = self.create_connection_gene(conn_to_split.input_node_id, new_node.node_id, 1)
        new_out_conn = self.create_connection_gene(new_node.node_id, conn_to_split.output_node_id, conn_to_split.weight)
        # print(f'Node mutated between nodes {conn_to_split.input_node_id} and {conn_to_split.output_node_id}')
        return True

    def attempt_topological_mutations(self):
        if rand.uniform(0, 1) < self._node_mutation_chance:
            if self.attempt_node_mutation():
                self._innovation_count += 1
        if rand.uniform(0, 1) < self._connection_mutation_chance:
            if self.attempt_connection_mutation():
                self._innovation_count += 1

    def mutate_connection_states(self):
        for conn in self._connection_genes:
            if rand.uniform(0, 1) < self._toggle_chance:
                conn.enabled = not conn.enabled
     
    def attempt_reenable_connection_mutation(self):
        for conn in self._connection_genes:
            if not conn.enabled:
                conn.enabled = True
                return True

        return False

    def add_and_mutate_connection_gene(self, gene):        
        new_gene = ConnectionGene(gene.innovation_id, self, gene.input_node_id, gene.output_node_id, gene.weight, gene.enabled)
        self._connection_genes.append(new_gene)
        self._connection_structures.add(new_gene.structure)

        if not new_gene.enabled:
            new_gene.enabled = rand.uniform(0, 1) < self._reenable_chance

        weight_mod = self.generate_weight_modifier()
        if rand.uniform(0, 1) < self._weight_mutation_chance:
            new_gene.weight += weight_mod
        elif rand.uniform(0, 1) < self._weight_cold_mutation_chance:
            new_gene.weight = weight_mod
            
        new_gene.weight = min(max(new_gene.weight, -self._weight_cap), self._weight_cap)

    def compatibilty(self, other):
        num_disjoint = 0
        num_excess = 0
        num_matching = 0
        total_weight_diff = 0
        larger_genotype_size = max(len(self.connection_genes), len(other.connection_genes))
        p1, p2 = 0, 0

        while p1 < len(self.connection_genes) or p2 < len(other.connection_genes):
            if p1 >= len(self.connection_genes):
                num_excess += 1
                p2 += 1
            elif p2 >= len(other.connection_genes):
                num_excess += 1 
                p1 += 1
            else:
                gene1 = self.connection_genes[p1]
                gene2 = other._connection_genes[p2]
                if gene1.innovation_id == gene2.innovation_id:
                    num_matching += 1
                    total_weight_diff += abs(gene1.weight - gene2.weight)
                    p1 += 1
                    p2 += 1
                elif gene1.innovation_id < gene2.innovation_id:
                    num_disjoint += 1
                    p1 += 1
                elif gene2.innovation_id < gene1.innovation_id:
                    num_disjoint += 1
                    p2 += 1
                else:
                    raise RuntimeError('Something went wrong')
            
        N = larger_genotype_size if larger_genotype_size >= 20 else 1

        return (self._excess_coeff * num_excess / N + 
                self._disjoint_coeff * num_disjoint / N + 
                self._weight_coeff * total_weight_diff / num_matching)

    def favored_crossover(self, other):
        num_disjoint = 0
        num_excess = 0
        num_matching = 0
        offspring_genotype = Genotype()
        offspring_genotype._node_genes = self.node_genes[:]
        p1, p2 = 0, 0
        while p1 < len(self.connection_genes) or p2 < len(other.connection_genes):
            if p1 >= len(self.connection_genes):
                num_excess += 1
                p2 += 1
            elif p2 >= len(other.connection_genes):
                num_excess += 1
                offspring_genotype.add_and_mutate_connection_gene(self.connection_genes[p1])
                p1 += 1
            else:
                gene1 = self.connection_genes[p1]
                gene2 = other.connection_genes[p2]
                if gene1.innovation_id == gene2.innovation_id:
                    num_matching += 1
                    # Must prevent inheriting a disabled connection without 
                    # inheriting the connections that replaced it
                    # NOTE: May be better to allow invalid networks and 
                    # to handle that in network activation method
                    # if not gene2.enabled and gene1.enabled:
                    #     chosen_gene = gene1
                    # else:
                    chosen_gene = rand.choice([gene1, gene2])

                    offspring_genotype.add_and_mutate_connection_gene(chosen_gene)
                    p1 += 1
                    p2 += 1
                elif gene1.innovation_id < gene2.innovation_id:
                    num_disjoint += 1
                    offspring_genotype.add_and_mutate_connection_gene(gene1)
                    p1 += 1
                elif gene2.innovation_id < gene1.innovation_id:
                    num_disjoint += 1
                    p2 += 1

        return offspring_genotype
                

    def equal_crossover(self, other):
        pass
        





    
        
