import random as rand
import time

from kypy_neat.genes import NodeType, gene_factory
from kypy_neat.utils.timer import timer


class Genotype:
    base_genotype = None
    _mutate_starting_topologies = False
    
    _weight_mutation_chance = 0.8 # chance a genotype's weights will be considered for mutation
    _weight_mutation_rate = 0.9 # chance for each individual weight to be perturbed
    _weight_cold_mutation_rate = 0.1 # chance for each individual weight to be completely replaced
    _end_genotype_threshold = 0.8 # point after which mutations should be more likely  
    _end_weight_mutation_rate = 0.9 # chance for each individual end weight to be perturbed
    _end_weight_cold_mutation_rate = 0.1 # chance for each individual end weight to be completely replaced

    # K. Stanley states mutation power should not exceed 5.0
    _weight_mut_power = 1.5
    _severe_weight_mut_chance = 0
    _severe_weight_mut_power = 5 
    _weight_cap = 8.0

    # K. Stanley states connection mutation chance should significantly exceed node mutation chance
    # He recommends 0.03 and 0.05, respectively, for small populations. These values seem to perform
    # well with this implementation
    _node_mutation_chance = 0.03 
    _connection_mutation_chance = 0.1

    _toggle_chance = 0 # chance a genotype's connections will be considered for toggling state
    _toggle_mutation_rate = 0.1  # chace for each individual connection to be toggles
    _reenable_chance = 0.01

    _excess_coeff = 1
    _disjoint_coeff = 1
    _weight_coeff = 0.4
    

    def __init__(self):
        self._node_genes = []
        self._connection_genes = []
        self._connection_structures = set()

    def generate_copy(self):
        genotype_copy = Genotype()
        for node_gene in self._node_genes:
            node_gene_copy = gene_factory.copy_node_gene(node_gene)
            genotype_copy._node_genes.append(node_gene_copy)

        for connection_gene in self._connection_genes:
            connection_gene_copy = gene_factory.copy_connection_gene(connection_gene)
            genotype_copy._connection_genes.append(connection_gene_copy)
            genotype_copy._connection_structures.add(connection_gene_copy.structure)
                                                                          
        return genotype_copy

    @classmethod
    def initialize_minimal_topology(cls, num_inputs, num_outputs):
        cls.base_genotype = Genotype()
        input_nodes = []
        output_nodes = []
        for _ in range(num_inputs):
            input_nodes.append(cls.base_genotype.create_node_gene(None, None, NodeType.INPUT))

        for _ in range(num_outputs):
            output_nodes.append(cls.base_genotype.create_node_gene(None, None, NodeType.OUTPUT))


        for i in range(len(input_nodes)):
            for j in range(len(output_nodes)):
                input_node = input_nodes[i]
                output_node = output_nodes[j]
                cls.base_genotype.create_connection_gene(input_node.innovation_id, output_node.innovation_id, cls.generate_weight_modifier())

    @classmethod
    def base_genotype_factory(cls):
        new_genotype = cls.base_genotype.generate_copy()
        if cls._mutate_starting_topologies:
            new_genotype.attempt_topological_mutations()
        new_genotype.mutate_weights()

        return new_genotype

    @property
    def node_genes(self):
        return self._node_genes

    @property
    def connection_genes(self):
        return self._connection_genes

    def structure_exists(self, structure):
        return structure in self._connection_structures


    # def mutate_weights(self):
    #     for conn in self._connection_genes:
    #         if rand.uniform(0, 1) < rate:
    #             if rand.uniform(0, 1) < 0.8:
    #                 conn.weight += self.generate_weight_modifier()
    #             else:
    #                 conn.weight = self.generate_weight_modifier()


    #         conn.weight = min(max(conn.weight, -self._weight_cap), self._weight_cap)

    def mutate_weights(self):
        for i, conn in enumerate(self._connection_genes):
            if i / len(self._connection_genes) < self._end_genotype_threshold:
                mutation_chance = self._end_weight_mutation_rate
                cold_mutation_chance = self._end_weight_cold_mutation_rate
     
            else:
                mutation_chance = self._weight_mutation_rate
                cold_mutation_chance = self._weight_cold_mutation_rate

            weight_mod = self.generate_weight_modifier()
            if rand.uniform(0, 1) < mutation_chance:
                conn.weight += weight_mod
            if rand.uniform(0, 1) < cold_mutation_chance:
                conn.weight = weight_mod
            
            conn.weight = min(max(conn.weight, -self._weight_cap), self._weight_cap)

    @classmethod
    def generate_weight_modifier(cls):
        if rand.uniform(0, 1) < cls._severe_weight_mut_chance:
            power = cls._severe_weight_mut_power
        else:
            power = cls._weight_mut_power

        return rand.uniform(-1, 1) * power

    def create_node_gene(self, input_node_id, output_node_id, node_type):
        node_gene = gene_factory.create_node_gene(self, input_node_id, output_node_id, node_type)
        self._node_genes.append(node_gene)
        return node_gene

    def create_connection_gene(self, input_node_id, output_node_id, weight, enabled=True):
        connection_gene = gene_factory.create_connection_gene(self, input_node_id, output_node_id, weight, enabled)
        self._connection_genes.append(connection_gene)
        self._connection_structures.add(connection_gene.structure)
        return connection_gene

    def attempt_connection_mutation(self):
        input_node = self._node_genes[rand.randint(0, len(self._node_genes) - 1)]  
        output_node = self._node_genes[rand.randint(0, len(self._node_genes) - 1)]
        structure = (input_node.innovation_id, output_node.innovation_id)
        weight = self.generate_weight_modifier()
        if not self.structure_exists(structure):
            self.create_connection_gene(input_node.innovation_id, output_node.innovation_id, weight)
            # print(f'Connection mutated between nodes {input_node.innovation_id} and {output_node.innovation_id}')
            return True

        return False

    def attempt_node_mutation(self):
        if not len(self._connection_genes):
            return False

        conn_to_split = self._connection_genes[rand.randint(0, len(self._connection_genes) - 1)]
        conn_to_split.enabled = False
        new_node = self.create_node_gene(conn_to_split.input_node_id, conn_to_split.output_node_id, NodeType.HIDDEN)
        new_input_conn = self.create_connection_gene(conn_to_split.input_node_id, new_node.innovation_id, 1)
        new_out_conn = self.create_connection_gene(new_node.innovation_id, conn_to_split.output_node_id, conn_to_split.weight)
        # print(f'Node mutated between nodes {conn_to_split.input_node_id} and {conn_to_split.output_node_id}')
        return True

    def attempt_topological_mutations(self):
        if rand.uniform(0, 1) < Genotype._node_mutation_chance:
            self.attempt_node_mutation()
        if rand.uniform(0, 1) < Genotype._connection_mutation_chance:
            self.attempt_connection_mutation()

    def attempt_all_mutations(self):
        if rand.uniform(0, 1) < Genotype._node_mutation_chance:
            self.attempt_node_mutation()
        if rand.uniform(0, 1) < Genotype._connection_mutation_chance:
            self.attempt_connection_mutation()
        if rand.uniform(0, 1) < Genotype._weight_mutation_chance:
            self.mutate_weights()
        if rand.uniform(0, 1) < Genotype._toggle_chance:
            self.mutate_connection_states()

    def mutate_connection_states(self):
        for conn in self._connection_genes:
            if rand.uniform(0, 1) < Genotype._toggle_mutation_rate:
                conn.enabled = not conn.enabled

    def attempt_reenable_connection_mutation(self):
        for conn in self._connection_genes:
            if not conn.enabled:
                conn.enabled = True
                return True

        return False

    def inherit_connection_gene(self, gene):  
        new_gene = gene_factory.create_connection_gene(self, gene.input_node_id, gene.output_node_id, gene.weight, gene.enabled)      
        self._connection_genes.append(new_gene)
        self._connection_structures.add(new_gene.structure)

        if not new_gene.enabled:
            new_gene.enabled = rand.uniform(0, 1) < Genotype._reenable_chance

    def add_and_mutate_connection_gene(self, gene):  
        new_gene = gene_factory.create_connection_gene(self, gene.input_node_id, gene.output_node_id, gene.weight, gene.enabled)      
        self._connection_genes.append(new_gene)
        self._connection_structures.add(new_gene.structure)

        if not new_gene.enabled:
            new_gene.enabled = rand.uniform(0, 1) < Genotype._reenable_chance

        weight_mod = self.generate_weight_modifier()
        if rand.uniform(0, 1) < Genotype._weight_mutation_chance:
            new_gene.weight += weight_mod
        elif rand.uniform(0, 1) < Genotype._weight_cold_mutation_chance:
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
                offspring_genotype.inherit_connection_gene(self.connection_genes[p1])
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
                    offspring_genotype.inherit_connection_gene(chosen_gene)
                    offspring_genotype._connection_genes[-1].weight = (gene1.weight + gene2.weight) / 2 # NOTE under test
                    p1 += 1
                    p2 += 1
                elif gene1.innovation_id < gene2.innovation_id:
                    num_disjoint += 1
                    offspring_genotype.inherit_connection_gene(gene1)
                    p1 += 1
                elif gene2.innovation_id < gene1.innovation_id:
                    num_disjoint += 1
                    p2 += 1

        # offspring_genotype.attempt_topological_mutations() # NOTE under test
        offspring_genotype.attempt_all_mutations()

        return offspring_genotype
                
    def copy_and_mutate(self):
        genotype_copy = self.generate_copy()
        genotype_copy.attempt_all_mutations()
        return genotype_copy

    def equal_crossover(self, other):
        pass
        





    
        
