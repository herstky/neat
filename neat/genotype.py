import random as rand

from neat.genes import NodeType, gene_factory
from neat.phenotype import Phenotype

class Genotype:
    base_genotype = None
    mutate_starting_topologies = False
    allow_recurrence = True
    
    starting_weight_variance = 1
    genotype_mutation_chance = 0.8
    stable_weight_mutation_chance = 0.9
    stable_weight_cold_mutation_chance = 0.1
    stable_gene_threshold = 0.8 
    unstable_weight_mutation_chance = 0.9
    unstable_weight_cold_mutation_chance = 0.1
    severe_weight_mutation_chance = 0.01
    weight_mutation_power = 5
    severe_weight_mutation_power = 5 
    weight_cap = 8.0

    node_mutation_chance = 0.03
    connection_mutation_chance = 0.1

    toggle_connection_chance = 0.1 # chance a genotype's connections will be considered for toggling state
    toggle_mutation_rate = 0.1  # chace for each individual connection to be toggled
    reenable_connection_chance = 0.2

    excess_coeff = 1
    disjoint_coeff = 1
    weight_coeff = 0.4
    
    def __init__(self):
        self._node_genes = []
        self._node_structures = set()
        self._connection_genes = []
        self._connection_structures = set()
    
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
        for node_gene in self._node_genes:
            genotype_copy._copy_node_gene(node_gene)
        for connection_gene in self._connection_genes:
            genotype_copy._copy_connection_gene(connection_gene)                      
        return genotype_copy

    def _copy_node_gene(self, node_gene):
        node_gene_copy = gene_factory.copy_node_gene(node_gene)
        self.add_node_gene(node_gene_copy)

    def add_node_gene(self, node_gene):
        idx = 0
        for existing_gene in self._node_genes:
            if node_gene.innovation_id > existing_gene.innovation_id:
                idx += 1
        self._node_genes.insert(idx, node_gene)
        self._node_structures.add(node_gene.structure)

    def _copy_connection_gene(self, connection_gene):
        connection_gene_copy = gene_factory.copy_connection_gene(connection_gene)
        self.add_connection_gene(connection_gene_copy)

    def add_connection_gene(self, connection_gene):
        idx = 0
        for existing_gene in self._connection_genes:
            if connection_gene.innovation_id > existing_gene.innovation_id:
                idx += 1
        self._connection_genes.insert(idx, connection_gene)
        self._connection_structures.add(connection_gene.structure)

    @classmethod
    def generate_mutated_base_genotype_copy(cls):
        genotype_copy = cls.base_genotype.generate_copy()
        if cls.mutate_starting_topologies:
            genotype_copy.attempt_topological_mutations()
        genotype_copy.mutate_weights()

        return genotype_copy

    def mutate_weights(self):
        for i, conn in enumerate(self._connection_genes):
            stable = not self._exceeds_stable_genes(i)
            self._mutate_weight(conn, stable)

    def _mutate_weight(self, conn, stable):
        if stable:
            self._mutate_stabe_gene(conn)
        else:
            self._mutate_unstable_gene(conn)
        self._cap_connection_weight

    def _exceeds_stable_genes(self, index):
        return index / len(self._connection_genes) < self.stable_gene_threshold

    def _mutate_unstable_gene(self, conn):
        self._attempt_weight_mutation(conn, self.unstable_weight_mutation_chance)
        self._attempt_cold_weight_mutation(conn, self.unstable_weight_cold_mutation_chance)

    def _mutate_stabe_gene(self, conn):
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
    def node_genes(self):
        return self._node_genes

    @property
    def connection_genes(self):
        return self._connection_genes

    @property
    def num_enabled_connection_genes(self):
        return len([gene for gene in self.connection_genes if gene.enabled])

    def node_structure_exists(self, structure):
        return structure in self._node_structures

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

    def create_connection_gene(self, input_node_id, output_node_id, weight, enabled=True):
        connection_gene = gene_factory.create_connection_gene(self, input_node_id, output_node_id, weight, enabled)
        self.add_connection_gene(connection_gene)
        return connection_gene

    def _attempt_node_mutation(self):
        connection_to_split = self._get_connection_to_split()
        if connection_to_split is None:
            return False
        self._build_node_mutation(connection_to_split)
        return True

    def _get_connection_to_split(self):
        conn_candidates = []
        for conn in self._connection_genes:
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
        connection_structure_to_add = self._get_connection_structure_to_add()
        if connection_structure_to_add is None:
            return False
        self._build_connection_mutation(connection_structure_to_add)
        return True

    def _get_connection_structure_to_add(self):
        structure_candidates = []
        for input_node in self._node_genes:
            for output_node in self._node_genes:
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
        return structure in self._connection_structures

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
        if self._event_occurs(Genotype.node_mutation_chance):
            self._attempt_node_mutation()
        if self._event_occurs(Genotype.connection_mutation_chance):
            self._attempt_connection_mutation()

    def attempt_all_mutations(self):
        if self._event_occurs(Genotype.node_mutation_chance):
            self._attempt_node_mutation()
        if self._event_occurs(Genotype.connection_mutation_chance):
            self._attempt_connection_mutation()
        if self._event_occurs(Genotype.genotype_mutation_chance):
            self.mutate_weights()
        if self._event_occurs(Genotype.toggle_connection_chance):
            self.mutate_connection_states()

    def mutate_connection_states(self):
        for conn in self._connection_genes:
            if self._event_occurs(Genotype.toggle_mutation_rate):
                conn.enabled = not conn.enabled

    def attempt_reenable_connection_mutation(self):
        for conn in self._connection_genes:
            if not conn.enabled:
                conn.enabled = True
                return True
        return False

    def inherit_connection_gene(self, gene):  
        connection_gene = gene_factory.create_connection_gene(self, gene.input_node_id, gene.output_node_id, gene.weight, gene.enabled)      
        self.add_connection_gene(connection_gene)

        if not connection_gene.enabled:
            connection_gene.enabled = self._event_occurs(Genotype.reenable_connection_chance)

    def add_and_mutate_connection_gene(self, gene):  
        connection_gene = gene_factory.create_connection_gene(self, gene.input_node_id, gene.output_node_id, gene.weight, gene.enabled)      
        self.add_connection_gene(connection_gene)

        if not connection_gene.enabled:
            connection_gene.enabled = self._event_occurs(Genotype.reenable_connection_chance)

        weight_delta = self._generate_weight_modifier()
        if self._event_occurs(Genotype.genotype_mutation_chance):
            connection_gene.weight += weight_delta
        elif self._event_occurs(Genotype._weight_cold_mutation_chance):
            connection_gene.weight = weight_delta
            
        connection_gene.weight = min(max(connection_gene.weight, -self.weight_cap), self.weight_cap)

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

        return (self.excess_coeff * num_excess / N + 
                self.disjoint_coeff * num_disjoint / N + 
                self.weight_coeff * total_weight_diff / num_matching)

    def crossover_genotypes(self, other):
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
                    chosen_gene = rand.choice([gene1, gene2])
                    offspring_genotype.inherit_connection_gene(chosen_gene)
                    offspring_genotype._connection_genes[-1].weight = (gene1.weight + gene2.weight) / 2
                    p1 += 1
                    p2 += 1
                elif gene1.innovation_id < gene2.innovation_id:
                    num_disjoint += 1
                    offspring_genotype.inherit_connection_gene(gene1)
                    p1 += 1
                elif gene2.innovation_id < gene1.innovation_id:
                    num_disjoint += 1
                    p2 += 1

        offspring_genotype.attempt_all_mutations()

        return offspring_genotype
                
    def copy_and_mutate(self):
        genotype_copy = self.generate_copy()
        genotype_copy.attempt_all_mutations()
        return genotype_copy
