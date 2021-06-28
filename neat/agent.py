from neat.utils.math import sigmoid

class Agent:
    _agents_created = 0
    _agent_count = 0
    def __init__(self, phenotype):
        Agent._agents_created += 1
        Agent._agent_count += 1
        self._agent_id = Agent._agents_created
        self._phenotype = phenotype
        self.age = 0
        self.error_sum = 0
        self.classification_error = 0
        self.fitness = 0
        self._killed = False

    @staticmethod
    def agents_created():
        return Agent._agents_created

    @property
    def innovation_count(self):
        return self.genotype.innovation_count

    @property
    def adjusted_fitness(self):
        return pow(self.fitness, 2)

    @property
    def agent_id(self):
        return self._agent_id

    @property
    def outputs(self):
        return [node.activation for node in self._phenotype.output_nodes]

    @property
    def expired(self):
        return self._killed

    def activate_network(self, inputs):
        return self._phenotype.evaluate_network(inputs)

    def calculate_fitness(self, *args, **kwargs):
        pass

    @property
    def phenotype(self):
        return self._phenotype

    @property
    def genotype(self):
        return self._phenotype.genotype

    def kill(self):
        Agent._agent_count -= 1
        self._killed = True