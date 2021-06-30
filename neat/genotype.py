import random as rand

from neat.genes import Gene, NodeType, gene_factory
from neat.phenotype import Phenotype
from neat.gene_set_list import GeneSetList

class Genotype:
    base_genotype = None
    mutate_starting_topologies = False
    allow_recurrence = False
    
    starting_weight_variance = 1
    genotype_mutation_chance = 0.8
    stable_weight_mutation_chance = 0.7
    stable_weight_cold_mutation_chance = 0.05
    stable_gene_threshold = 0.8 
    unstable_weight_mutation_chance = 0.7
    unstable_weight_cold_mutation_chance = 0.05
    severe_weight_mutation_chance = 0.1
    weight_mutation_power = 5
    severe_weight_mutation_power = 5
    weight_cap = 8.0

    node_mutation_chance = 0.02
    connection_mutation_chance = 0.05

    toggle_connection_chance = 0.01
    reenable_connection_chance = 0.2

    excess_coeff = 1
    disjoint_coeff = 1
    weight_coeff = 0.4
    
    def __init__(self):
        self._node_gene_set_list = GeneSetList()
        self._connection_gene_set_list = GeneSetList()
    
    @classmethod
    def initialize(cls, num_inputs, num_outputs):
        cls.base_genotype = cls._generate_base_genotype(num_inputs, num_outputs)

    @classmethod
    def _generate_base_genotype(cls, num_inputs, num_outputs):
        base_genotype = Genotype()
        base_genotype._build_minimal_topology(num_inputs, num_outputs)
        return base_genotype

    def _build_minimal_topology(self, num_inputs, num_outputs):
        input_nodes = [self._create_input_node() for _ in range(num_inputs)]
        output_nodes = [self._create_output_node() for _ in range(num_outputs)]
        self._build_connections(input_nodes, output_nodes)

    def _build_connections(self, input_nodes, output_nodes):
        for i in range(len(input_nodes)):
            for j in range(len(output_nodes)):
                input_node = input_nodes[i]
                output_node = output_nodes[j]
                self.create_connection_gene(
                    input_node.innovation_id, 
                    output_node.innovation_id, 
                    self._generate_starting_weight())
    
    def create_connection_gene(self, input_node_id, output_node_id, weight, enabled=True):
        connection_gene = gene_factory.create_connection_gene(self, input_node_id, output_node_id, weight, enabled)
        self.add_connection_gene(connection_gene)
        return connection_gene

    @classmethod
    def _generate_starting_weight(cls):
        return rand.uniform(-1, 1) * cls.starting_weight_variance  

    @classmethod
    def _generate_weight_modifier(cls):
        if cls._event_occurs(cls.severe_weight_mutation_chance):
            return cls._generate_severe_weight_modifier()
        else:
            return cls._generate_normal_weight_modifier()

    @staticmethod
    def _event_occurs(chance):
        return rand.uniform(0, 1) < chance

    @classmethod
    def _generate_normal_weight_modifier(cls):
        return rand.uniform(-1, 1) * cls.weight_mutation_power

    @classmethod
    def _generate_severe_weight_modifier(cls):
        return rand.uniform(-1, 1) * cls.severe_weight_mutation_power

    def generate_copy(self):
        genotype_copy = Genotype()
        self._copy_node_genes_to(genotype_copy)
        self._copy_connection_genes_to(genotype_copy)                      
        return genotype_copy

    def _copy_node_genes_to(self, genotype_copy):
        for node_gene in self.node_genes:
            genotype_copy._copy_node_gene(node_gene)

    @property
    def node_genes(self):
        return self._node_gene_set_list.genes

    def _copy_connection_genes_to(self, genotype_copy):
        for connection_gene in self.connection_genes:
            genotype_copy._copy_connection_gene(connection_gene)

    @property
    def connection_genes(self):
        return self._connection_gene_set_list.genes

    def _copy_node_gene(self, node_gene):
        node_gene_copy = gene_factory.copy_node_gene(node_gene)
        self.add_node_gene(node_gene_copy)

    def add_node_gene(self, node_gene):
        self._node_gene_set_list.add_gene(node_gene)

    def _copy_connection_gene(self, connection_gene):
        connection_gene_copy = gene_factory.copy_connection_gene(connection_gene)
        self.add_connection_gene(connection_gene_copy)

    def add_connection_gene(self, connection_gene):
        self._connection_gene_set_list.add_gene(connection_gene)

    @classmethod
    def generate_mutated_base_genotype_copy(cls):
        genotype_copy = cls.base_genotype.generate_copy()
        if cls.mutate_starting_topologies:
            genotype_copy.attempt_topological_mutations()
        genotype_copy.mutate_weights()

        return genotype_copy

    def mutate_weights(self):
        for i, conn in enumerate(self.connection_genes):
            stable = not self._exceeds_stable_genes(i)
            self._attempt_mutate_weight(conn, stable)

    def _attempt_mutate_weight(self, conn, stable=True):
        if stable:
            self._attempt_mutate_stable_connection(conn)
        else:
            self._attempt_mutate_unstable_connection(conn)
        self._cap_connection_weight(conn)

    def _exceeds_stable_genes(self, index):
        return index / self.num_connection_genes < self.stable_gene_threshold

    @property
    def num_connection_genes(self):
        return len(self.connection_genes)

    def _attempt_mutate_unstable_connection(self, conn):
        self._attempt_weight_mutation(conn, self.unstable_weight_mutation_chance)
        self._attempt_cold_weight_mutation(conn, self.unstable_weight_cold_mutation_chance)

    def _attempt_mutate_stable_connection(self, conn):
        self._attempt_weight_mutation(conn, self.stable_weight_mutation_chance)
        self._attempt_cold_weight_mutation(conn, self.stable_weight_cold_mutation_chance)

    def _attempt_weight_mutation(self, conn, chance):
        if self._event_occurs(chance):
            conn.weight += self._generate_weight_modifier()

    def _attempt_cold_weight_mutation(self, conn, chance):
        if self._event_occurs(chance):
            conn.weight = self._generate_weight_modifier()

    def _cap_connection_weight(self, conn):
        conn.weight = min(max(conn.weight, -self.weight_cap), self.weight_cap)

    @property
    def num_enabled_connection_genes(self):
        return len([gene for gene in self.connection_genes if gene.enabled])

    def node_structure_exists(self, structure):
        return self._node_gene_set_list.contains_structure(structure)

    def _create_input_node(self):
        return self._create_node_gene(None, None, NodeType.INPUT)

    def _create_output_node(self):
        return self._create_node_gene(None, None, NodeType.OUTPUT)

    def _create_hidden_node(self, input_node_id, output_node_id):
        return self._create_node_gene(input_node_id, output_node_id, NodeType.HIDDEN)

    def _create_node_gene(self, input_node_id, output_node_id, node_type):
        node_gene = gene_factory.create_node_gene(self, input_node_id, output_node_id, node_type)
        self.add_node_gene(node_gene)
        return node_gene

    def _attempt_node_mutation(self):
        if not self._event_occurs(Genotype.node_mutation_chance):
            return
        connection_to_split = self._get_connection_to_split()
        if connection_to_split:
            self._build_node_mutation(connection_to_split)

    def _get_connection_to_split(self):
        conn_candidates = []
        for conn in self.connection_genes:
            if not self.node_structure_exists(conn.structure):
                conn_candidates.append(conn)
        if not len(conn_candidates):
            return None
        else:
            return rand.choice(conn_candidates)

    def _build_node_mutation(self, connection_to_split):
        new_node = self._create_hidden_node(
            connection_to_split.input_node_id, 
            connection_to_split.output_node_id)
        connection_to_split.enabled = False
        self.create_connection_gene(
            connection_to_split.input_node_id, 
            new_node.innovation_id,
            1)
        self.create_connection_gene(
            new_node.innovation_id, 
            connection_to_split.output_node_id, 
            connection_to_split.weight)

    def _attempt_connection_mutation(self):
        if not self._event_occurs(Genotype.connection_mutation_chance):
            return
        connection_structure_to_add = self._get_connection_structure_to_add()
        if connection_structure_to_add:
            self._build_connection_mutation(connection_structure_to_add)

    def _get_connection_structure_to_add(self):
        structure_candidates = []
        for input_node in self.node_genes:
            for output_node in self.node_genes:
                if self._connection_mutation_is_valid(input_node, output_node):
                    structure = (input_node.innovation_id, output_node.innovation_id)
                    structure_candidates.append(structure)
                
        if not len(structure_candidates):
            return None
        else:
            return rand.choice(structure_candidates)

    def _connection_mutation_is_valid(self, input_node, output_node):
        structure = (input_node.innovation_id, output_node.innovation_id)
        return (not self._connection_structure_exists(structure) and
            not self._bridges_input_nodes(input_node, output_node) and
            not self._bridges_output_nodes(input_node, output_node) and
            self._recurrency_is_valid(input_node.innovation_id, output_node.innovation_id))
    
    def _connection_structure_exists(self, structure):
        return self._connection_gene_set_list.contains_structure(structure)

    def _bridges_input_nodes(self, input_node, output_node):
        return (input_node.node_type is NodeType.INPUT 
            and output_node.node_type is NodeType.INPUT 
            and input_node is not output_node)

    def _bridges_output_nodes(self, input_node, output_node):
        return (input_node.node_type is NodeType.OUTPUT 
            and output_node.node_type is NodeType.OUTPUT 
            and input_node is not output_node)

    def _recurrency_is_valid(self, input_node_id, output_node_id):
        return not self._is_recurrent(input_node_id, output_node_id) or Genotype.allow_recurrence

    def _is_recurrent(self, input_node_id, output_node_id):
        test_phenotype = Phenotype(self)
        input_node = test_phenotype.get_node(input_node_id)
        output_node = test_phenotype.get_node(output_node_id)

        visited = set()
        stack = [output_node]
        while len(stack):
            node = stack.pop()
            if node not in visited:
                if node is input_node:
                    return True
                visited.add(node)
                for output_connnection in node.output_connections:
                    stack.append(output_connnection.output_node)

        return False

    def _build_connection_mutation(self, connection_structure):
        input_node_id, output_node_id = connection_structure
        weight = self._generate_starting_weight()
        self.create_connection_gene(input_node_id, output_node_id, weight)

    def attempt_topological_mutations(self):
        self._attempt_node_mutation()
        self._attempt_connection_mutation()

    def attempt_all_mutations(self):
        self._attempt_node_mutation()
        self._attempt_connection_mutation()
        self._attempt_mutate_all_weights()
        self._attempt_mutate_connection_states()

    def _attempt_mutate_all_weights(self):
        if self._event_occurs(Genotype.genotype_mutation_chance):
            self.mutate_weights()
        
    def _attempt_mutate_connection_states(self):
        for conn in self.connection_genes:
            self._attempt_toggle_connection(conn)

    def _attempt_toggle_connection(self, conn):
        if self._event_occurs(Genotype.toggle_connection_chance):
            conn.enabled = not conn.enabled

    def _inherit_connection_gene(self, gene):  
        connection_gene = gene_factory.create_connection_gene(self, gene.input_node_id, gene.output_node_id, gene.weight, gene.enabled)      
        self.add_connection_gene(connection_gene)
        self._attempt_reenable_connection(connection_gene)

    def _attempt_reenable_connection(self, conn):
        if not conn.enabled:
            conn.enabled = self._event_occurs(Genotype.reenable_connection_chance)

    def _add_and_mutate_connection_gene(self, gene):  
        connection_gene = gene_factory.create_connection_gene(self, gene.input_node_id, gene.output_node_id, gene.weight, gene.enabled)      
        self.add_connection_gene(connection_gene)
        self._attempt_reenable_connection(connection_gene)
        self._attempt_mutate_weight(connection_gene)
            
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
                gene2 = other.connection_genes[p2]
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
                    raise RuntimeError(f'Invalid argument {other}')
            
        N = larger_genotype_size if larger_genotype_size >= 20 else 1

        return (self.excess_coeff * num_excess / N + 
                self.disjoint_coeff * num_disjoint / N + 
                self.weight_coeff * total_weight_diff / num_matching)

    def crossover_genotypes(self, other):
        num_disjoint = 0
        num_excess = 0
        num_matching = 0
        offspring_genotype = Genotype()
        offspring_genotype._node_gene_set_list = self._generate_node_gene_set_list_copy()
        p1, p2 = 0, 0
        while p1 < len(self.connection_genes) or p2 < len(other.connection_genes):
            if p1 >= len(self.connection_genes):
                num_excess += 1
                p2 += 1
            elif p2 >= len(other.connection_genes):
                num_excess += 1
                offspring_genotype._inherit_connection_gene(self.connection_genes[p1])
                p1 += 1
            else:
                gene1 = self.connection_genes[p1]
                gene2 = other.connection_genes[p2]
                if gene1.innovation_id == gene2.innovation_id:
                    num_matching += 1
                    chosen_gene = rand.choice([gene1, gene2])
                    offspring_genotype._inherit_connection_gene(chosen_gene)
                    offspring_genotype.connection_genes[-1].weight = (gene1.weight + gene2.weight) / 2
                    p1 += 1
                    p2 += 1
                elif gene1.innovation_id < gene2.innovation_id:
                    num_disjoint += 1
                    offspring_genotype._inherit_connection_gene(gene1)
                    p1 += 1
                elif gene2.innovation_id < gene1.innovation_id:
                    num_disjoint += 1
                    p2 += 1

        offspring_genotype.attempt_all_mutations()

        return offspring_genotype

    def _generate_node_gene_set_list_copy(self):
        return self._node_gene_set_list.generate_copy()

    def copy_and_mutate(self):
        genotype_copy = self.generate_copy()
        genotype_copy.attempt_all_mutations()
        return genotype_copy
