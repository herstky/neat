import random as rand

from neat.phenotype import Phenotype
from neat.agent import Agent
from neat.utils.timer import timer


class Species:
    _species_created = 0
    _species_count = 0
    target_species_count = 10
    compatibility_threshold = 6.0
    compatibility_mod = 0.3
    control_species_count = True
    cull_fraction = 0.8
    stagnation_limit = 10
    stagnation_penalty = 0.2

    def __init__(self, representative):
        Species._species_created += 1
        Species._species_count += 1
        self._species_id = Species._species_created
        self._representative_genotype = representative.genotype.generate_copy()
        self._agents = []
        self._agent_set = set()
        self.results = {}
        self._age = 0
        self.expired = False
        self._stagnation_duration = 0
        self._best_fitness = float('-inf')

    @property
    def age(self):
        return self._age

    @classmethod
    def control_species_count(cls):
        if not cls.control_species_count:
            return

        if cls._species_count > cls.target_species_count:
            cls.compatibility_threshold += cls.compatibility_mod
        elif cls._species_count < cls.target_species_count:
            cls.compatibility_threshold -= cls.compatibility_mod

        cls.compatibility_threshold = max(cls.compatibility_mod, cls.compatibility_threshold)

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

    def annihilate(self):
        Species._species_count -= 1

    @property
    def living_agents(self):
        return [agent for agent in self._agents if not agent.expired]

    @property
    def num_alive(self):
        return len(self.living_agents)

    @property
    def min_culling_age(self):
        return self._min_culling_age

    @property
    def stagnated(self):
        return self._stagnation_duration >= Species.stagnation_limit

    def compatible(self, agent):
        return agent.genotype.compatibilty(self.representative_genotype) < Species.compatibility_threshold

    def in_species(self, agent):
        return agent in self._agent_set

    def advance(self):
        self._age += 1
        progressed = False
        for agent in self._agents:
            if agent.fitness > self._best_fitness:
                self._best_fitness = agent.fitness
                progressed = True
        
        if progressed:
            self._stagnation_duration = 0
        else:
            self._stagnation_duration += 1

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
        
        penalty = Species.stagnation_penalty if self.stagnated else 0

        return agent.adjusted_fitness * (1 - penalty) / self.count

    @property
    def average_fitness(self):
        return self.total_fitness / self.count

    @property
    def average_adjusted_fitness(self):
        return self.total_adjusted_fitness / self.count

    @property
    def max_fitness(self):
        return ranked_agents()[0].fitness

    @property
    def max_shared_fitness(self):
        return self.fitness_share(self.ranked_agents()[0])

    @property
    def min_shared_fitness(self):
        return self.fitness_share(self.ranked_agents()[-1])

    @property
    def total_fitness(self):
        return sum([agent.fitness for agent in self._agents])

    @property
    def total_shared_fitness(self):
        return sum([self.fitness_share(agent) for agent in self._agents])

    @property
    def total_adjusted_fitness(self):
        return sum([agent.adjusted_fitness for agent in self._agents])

    @property
    def total_living_shared_fitness(self):
        return sum([self.fitness_share(agent) for agent in self.living_agents])

    @property
    def average_shared_fitness(self):
        return self.total_shared_fitness / self.count

    def ranked_agents(self, descending=True):
        return sorted(self._agents, key=lambda agent: self.fitness_share(agent), reverse=descending)
    
    @property
    def champion(self):
        return self.ranked_agents()[0]

    def mutate(self):
        for agent in self._agents:
            if agent is self.champion:
                continue

            agent.phenotype.genotype.mutate()

    def cull(self):
        ranked_agents = self.ranked_agents(False)
        num_to_cull = int(Species.cull_fraction * self.count)

        for i in range(num_to_cull):
            ranked_agents[i].kill()

    @staticmethod
    def generate_offspring(parent1, parent2=None):
        if parent2 is None:
            genotype = parent1.genotype.copy_and_mutate()
        elif parent1.adjusted_fitness == parent2.adjusted_fitness: # adjusted_fitness must then be equal for both parents
            if rand.uniform(0, 1) > 0.5:
                genotype = parent1.genotype.favored_crossover(parent2.genotype)
            else:
                genotype = parent2.genotype.favored_crossover(parent1.genotype)
        else:
            genotype = parent1.genotype.favored_crossover(parent2.genotype)

        phenotype = Phenotype(genotype)
        return Agent(phenotype)
    
    @staticmethod
    def generate_clone(agent):
        genotype = agent.genotype.generate_copy()
        phenotype = Phenotype(genotype)
        return Agent(phenotype)

    def breed(self, offspring_share):
        offspring = []
        ranked_agents = self.ranked_agents()
        total_shared_fitness = self.total_living_shared_fitness
        living = self.num_alive

        if living == 0 or total_shared_fitness <= 0:
            return offspring

        # champ gets bonus offspring
        if len(self._agents) > 3:
            champ = ranked_agents[0]
            num_champ_clones = 1
            for _ in range(num_champ_clones):
                offspring.append(Species.generate_clone(champ))

            remaining_offspring_share = offspring_share - num_champ_clones
        else:
            remaining_offspring_share = offspring_share

        i = 0
        while i < living:
            if i + 1 > living - 1:
                parent = ranked_agents[i]
                num_offspring = round(remaining_offspring_share * self.fitness_share(parent) / total_shared_fitness)
                for _ in range(num_offspring):
                    offspring.append(Species.generate_offspring(parent))
                i += 1
            else:
                parent1 = ranked_agents[i]
                parent2 = ranked_agents[i + 1]
                num_offspring = round(remaining_offspring_share * (self.fitness_share(parent1) + self.fitness_share(parent2)) / total_shared_fitness) 
                for _ in range(num_offspring):
                    offspring.append(Species.generate_offspring(parent1, parent2))
                i += 2

        # any remaining offspring go to champ
        while len(offspring) < round(offspring_share):
            champ = ranked_agents[0]
            offspring.append(Species.generate_offspring(champ))
        
        return offspring

    def reset(self):
        self._agents = []
        self._agent_set = set()

    # def record_results(self):
    #     ranked_agents = self.ranked_agents()
    #     self._champion = ranked_agents[0]
    #     total_shared_fitness = self.total_shared_fitness
    #     self.results['size'] = self.count
    #     self.results['tot_shared_fitness'] = self.total_shared_fitness
    #     self.results['avg_shared_fitness'] = self.average_shared_fitness
    #     self.results['max_shared_fitness'] = self.fitness_share(ranked_agents[0])
    #     self.results['min_shared_fitness'] = self.fitness_share(ranked_agents[-1])


