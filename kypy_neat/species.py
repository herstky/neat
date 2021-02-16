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
        self.total_fitness_record = []
        self.age = 0
        self._min_culling_age = 3 # number of generations before a species will be eligible for annihilation
        self.expired = False
        self._cull_fraction = 0.3 # fraction of species to kill each generation
        self._population_floor = 0
        self._breed_fraction = 0.6
        self._base_offspring_count = 3

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

    def ranked_agents(self, descending=True):
        return sorted(self._agents, key=lambda agent: self.fitness_share(agent), reverse=descending)
    
    @property
    def champion(self):
        return self.ranked_agents()[0]

    def cull(self):
        ranked_agents = self.ranked_agents(False)
        # print(f'Species {self._species_id} starting population: {len(self._agents)}')
        if len(ranked_agents) > self._population_floor:
            cull_pop = max(1, int(len(ranked_agents) * self._cull_fraction))
        else:
            cull_pop = 0
        # print(f'Population to cull: {cull_pop}')
        for i in range(cull_pop):
            ranked_agents[i].kill()

        agents_str = 'Agents: '
        for agent in ranked_agents:
            agents_str += f'{agent.agent_id} '
            if agent.expired:
                self.remove(agent)
        
        # print(agents_str)
        # print(f'Species {self._species_id} ending population: {len(self._agents)}')
        # print()

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
        else: # adjusted_fitness must be equal for both parents
            if rand.uniform(0, 1) > 0.5:
                genotype = parent1.genotype.favored_crossover(parent2.genotype)
            else:
                genotype = parent2.genotype.favored_crossover(parent1.genotype)

        genotype.attempt_topological_mutations()

        phenotype = Phenotype(genotype)
        return Agent(phenotype)

    def breed(self, population, interspecies_rank_multiplier):
        offspring = []
        if self.total_shared_fitness <= 0:
            return offspring
            
        if len(self._agents) == 1:
            parent = self._agents[0]
            normalized_fitness = self.fitness_share(parent) / self.fitness_share(self.champion)
            intraspecies_rank_multiplier = 1
            if parent is population.generation_champion:
                intraspecies_rank_multiplier *= 5
            max_offspring = int(3 * self._base_offspring_count * normalized_fitness * interspecies_rank_multiplier * intraspecies_rank_multiplier)
            min_offspring = int(max_offspring / 2)
            num_offspring = rand.randint(min_offspring, max_offspring)
            for _ in range(num_offspring):
                offspring.append(self.generate_offspring(parent, parent))
            return offspring

        ranked_agents = self.ranked_agents()
        breeding_population_size = len(self._agents) * self._breed_fraction
        i = 0
        while i + 1 < breeding_population_size:
            parent1 = ranked_agents[i]
            parent2 = ranked_agents[i + 1]
            mate_fitness = (self.fitness_share(parent1) + self.fitness_share(parent2)) / 2
            normalized_fitness = mate_fitness / self.fitness_share(self.champion)
            intraspecies_rank_multiplier = 1
            if population.generation_champion in (parent1, parent2):
                intraspecies_rank_multiplier *= 5
            if i == 0:
                intraspecies_rank_multiplier *= 2
            elif i < 2:
                intraspecies_rank_multiplier *= 1.3
            elif i > 10:
                intraspecies_rank_multiplier *= 0.5

            max_offspring = int(self._base_offspring_count * normalized_fitness * interspecies_rank_multiplier * intraspecies_rank_multiplier)
            min_offspring = int(max_offspring / 2) 
            if max_offspring > 0:
                num_offspring = rand.randint(min_offspring, max_offspring)
                for _ in range(num_offspring):
                    offspring.append(self.generate_offspring(parent1, parent2))
            i += 2

        return offspring

    def reset(self):
        champion = self.champion
        self.total_fitness_record.append(self.total_shared_fitness)
        self._agents = []
        self._agent_set = set()

