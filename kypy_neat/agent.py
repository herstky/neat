import random as rand

from kypy_neat.utils.math import sigmoid


class Agent:
    _agent_count = 0
    def __init__(self, phenotype):
        Agent._agent_count += 1
        self._agent_id = Agent._agent_count
        self._phenotype = phenotype
        self.age = 0
        self._life_expectancy = 8
        self.error_sum = 0
        self.fitness = 0
        self._kill = False
        self._innovation_coeff = 0.5

    @property
    def innovation_count(self):
        return self.genotype.innovation_count

    @property
    def adjusted_fitness(self):
        # return self.fitness * (1 - sigmoid(self.age, 3 / 2, -self._life_expectancy)) * (1 + self._innovation_coeff * self.innovation_count)
        # return self.fitness * (1 + self._innovation_coeff * self.innovation_count)
        return self.fitness 

    @property
    def agent_id(self):
        return self._agent_id

    @property
    def outputs(self):
        return [node.activation for node in self._phenotype.output_nodes]

    @property
    def expired(self):
        aged_out = rand.uniform(0, 1) < sigmoid(self.age, 3 / 2, -self._life_expectancy)
        return aged_out or self._kill

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
        self._kill = True