import time

from kypy_neat.species import Species
from kypy_neat.genotype import Genotype
from kypy_neat.phenotype import Phenotype
from kypy_neat.agent import Agent
from kypy_neat.utils.timer import timer


class Population:
    def __init__(self):
        self._agents = []
        self._species = []
        self._compatibility_threshold = 3.0
        self._starting_population = 150
        self._cull_fraction = 0.2 # fraction of species to be annihilated each generation
        self._species_floor = 8 # minimum number of species that must exist before any are annihilated
        self._breed_fraction = 1 # fraction of species that will breed each round
        self._min_breeding_species = 4 # minimum number of species that will breed each round
        self._generation_champion = None

    @property
    def agents(self):
        return self._agents

    @property
    def species(self):
        return self._species

    @property
    def generation_champion(self):
        return self._generation_champion

    @property
    def count(self):
        return len(self._agents)

    @property
    def species_count(self):
        return len(self._species)

    def sorted_species(self):
        return sorted(self._species, key=lambda x: x.total_shared_fitness, reverse=True)

    def create_species(self, representative):
        new_species = Species(representative)
        new_species.add(representative)
        self._species.append(new_species)
        return new_species

    def age_population(self):
        for agent in self._agents:
            agent.age += 1

        for species in self._species:
            species.age += 1

    def initialize_population(self, num_inputs, num_outputs):
        Genotype.initialize_minimal_topology(num_inputs, num_outputs)
        for _ in range(self._starting_population):
            genotype = Genotype.base_genotype_factory()
            phenotype = Phenotype(genotype)
            self._agents.append(Agent(phenotype))

    def compatible(self, agent, species):
        return agent.genotype.compatibilty(species.representative_genotype) < self._compatibility_threshold

    def speciate(self):
        self.reset_species()

        for agent in self._agents:
            for species in self._species:
                if self.compatible(agent, species):
                    species.add(agent)
                    break
            else:
                self.create_species(agent)

    def kill_species_agents(self, species):
        agents = species.agents
        for agent in agents:
            agent.kill()

        self.cleanup_agents()

    def cull_species(self):
        for species in self._species:
            species.cull()

        sorted_species = self.sorted_species()
        sorted_species.reverse()
        if len(self._species) > self._species_floor:
            species_to_cull = int(len(self._species) * self._cull_fraction)
        else:
            species_to_cull = 0
        for i in range(species_to_cull):
            species = sorted_species[i]
            if species.age > species.min_culling_age:
                self.kill_species_agents(species)
                self._species.remove(species)

        species_copy = self._species[:]
        for species in species_copy:
            if species.extinct:
                self._species.remove(species)

    def reset_species(self):
        for species in self._species:
            species.reset()

    def mutate_species(self):
        for species in self._species:
            species.mutate()

    def breed_species(self):
        offspring = []
        sorted_species = self.sorted_species()
        breeding_pop = min(len(self._species), max(self._min_breeding_species, int(len(sorted_species) * self._breed_fraction)))
        for i in range(breeding_pop):
            interspecies_rank_multiplier = 1
            if i < 2:
                interspecies_rank_multiplier *= 2.0
            elif i < 4:
                interspecies_rank_multiplier *= 1.5
            elif i < 6:
                interspecies_rank_multiplier *= 1.2

            offspring += self._species[i].breed(self, interspecies_rank_multiplier)
        
        return offspring

    def prepare_generation(self):
        self._generation_champion = None
        self.cleanup_agents()
        self.speciate()
        pass

    def evaluate_agents(self, inputs, outputs):
        for agent in self._agents:
            agent.error_sum = 0
            for input_, expected_outputs in zip(inputs, outputs):
                net_outputs, net_error = agent.activate_network(input_)
                for net_output, expected_output in zip(net_outputs, expected_outputs):
                    agent.error_sum += abs(expected_output - net_output)
                agent.error_sum += net_error

            agent.fitness = pow(max(0, 4 - agent.error_sum), 2)
            if self._generation_champion:
                if agent.error_sum < self._generation_champion.error_sum:
                    self._generation_champion = agent
            else:
                self._generation_champion = agent

    def finish_generation(self):
        self.cull_species()
        offspring = self.breed_species()
        print(f'{len(offspring)} offspring generated')
        self.mutate_species()
        self.age_population()
        self._agents += offspring

    def cleanup_agents(self):
        agents_copy = self._agents[:]
        for agent in agents_copy:
            if agent.expired:
                self._agents.remove(agent)

