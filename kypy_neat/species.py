import random as rand

from kypy_neat.phenotype import Phenotype
from kypy_neat.agent import Agent
from kypy_neat.utils.timer import timer


class Species:
    _species_count = 0
    def __init__(self, representative):
        Species._species_count += 1
        self._species_id = Species._species_count
        self._representative_genotype = representative.genotype.generate_copy()
        self._agents = []
        self._agent_set = set()
        self.results = {}
        self.age = 0
        self._min_culling_age = 3 # number of generations before a species will be eligible for annihilation
        self.expired = False
        self._cull_fraction = 0.3 # fraction of species to kill each generation
        self._population_floor = 0
        self._breed_fraction = 0.6
        self._base_offspring_count = 3
        self._champion = None

    @property
    def representative_genotype(self):
        return self._representative_genotype

    @property
    def species_id(self):
        return self._species_id

    @property
    def agents(self):
        return tuple(self._agents)

    @property
    def extinct(self):
        return len(self._agents) == 0

    @property
    def min_culling_age(self):
        return self._min_culling_age

    def add(self, agent):
        if agent in self._agent_set:
            raise RuntimeError('Agent already in this species')

        self._agents.append(agent)
        self._agent_set.add(agent)

    def remove(self, agent):
        self._agents.remove(agent)
        self._agent_set.remove(agent)

    @property
    def count(self):
        return len(self._agents)

    def fitness_share(self, agent):
        if agent not in self._agent_set:
            raise RuntimeError('Agent not in this species')

        return agent.adjusted_fitness / self.count

    @property
    def average_fitness(self):
        return self.total_fitness / self.count

    @property
    def max_fitness(self):
        return ranked_agents()[0].fitness

    @property
    def max_shared_fitness(self):
        return self.fitness_share(ranked_agents()[0])

    @property
    def total_fitness(self):
        return sum([agent.fitness for agent in self._agents])

    @property
    def total_shared_fitness(self):
        return sum([self.fitness_share(agent) for agent in self._agents])

    @property
    def average_shared_fitness(self):
        return self.total_shared_fitness / self.count

    def ranked_agents(self, descending=True):
        return sorted(self._agents, key=lambda agent: self.fitness_share(agent), reverse=descending)
    
    @property
    def champion(self):
        return self._champion

    def cull(self):
        cull_threshold = self._cull_fraction * self.average_shared_fitness
        for agent in self._agents:
            if self.fitness_share(agent) < cull_threshold:
                agent.kill()

    def mutate(self):
        for agent in self._agents:
            if agent is self.champion:
                continue

            agent.phenotype.genotype.mutate()

    def generate_offspring(self, parent1, parent2):
        if parent1.adjusted_fitness < parent2.adjusted_fitness:
            raise RuntimeError('Unexpected adjusted_fitness pairing')

        if parent1.adjusted_fitness > parent2.adjusted_fitness:
            genotype = parent1.genotype.favored_crossover(parent2.genotype)
        else: # adjusted_fitness must then be equal for both parents
            if rand.uniform(0, 1) > 0.5:
                genotype = parent1.genotype.favored_crossover(parent2.genotype)
            else:
                genotype = parent2.genotype.favored_crossover(parent1.genotype)

        genotype.attempt_topological_mutations()

        phenotype = Phenotype(genotype)
        return Agent(phenotype)

    def breed(self, offspring_share):
        offspring = []
        ranked_agents = self.ranked_agents()
        total_shared_fitness = self.total_shared_fitness
        living = 0
        for agent in ranked_agents:
            if agent.expired:
                break
            else:
                living += 1

        if living == 0 or total_shared_fitness <= 0:
            return offspring

        i = 0
        while i < living:
            if i + 1 > living - 1:
                parent = ranked_agents[i]
                num_offspring = round(offspring_share * self.fitness_share(parent) / total_shared_fitness)
                for _ in range(num_offspring):
                    offspring.append(self.generate_offspring(parent, parent))
                i += 1
            else:
                parent1 = ranked_agents[i]
                parent2 = ranked_agents[i + 1]
                num_offspring = round(offspring_share * (self.fitness_share(parent1) + self.fitness_share(parent2)) / total_shared_fitness) 
                for _ in range(num_offspring):
                    offspring.append(self.generate_offspring(parent1, parent2))
                i += 2
        
        return offspring

    def reset(self):
        self._agents = []
        self._agent_set = set()

    def record_results(self):
        ranked_agents = self.ranked_agents()
        self._champion = ranked_agents[0]
        total_shared_fitness = self.total_shared_fitness
        self.results['size'] = self.count
        self.results['tot_shared_fitness'] = total_shared_fitness
        self.results['avg_shared_fitness'] = total_shared_fitness / self.count
        self.results['max_shared_fitness'] = self.fitness_share(ranked_agents[0])
        self.results['min_shared_fitness'] = self.fitness_share(ranked_agents[-1])


