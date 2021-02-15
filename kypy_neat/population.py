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
        self._cull_fraction = 0.1

    @property
    def agents(self):
        return self._agents

    @property
    def species(self):
        return self._species

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
        species_to_cull = int(len(self._species) * self._cull_fraction)
        for i in range(species_to_cull):
            species = sorted_species[i]
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
        for i, species in enumerate(sorted_species):
            multiplier = 1 - i / len(sorted_species)
            if i < 2:
                multiplier *= 3
            elif i < 4:
                multiplier *= 1.5
            elif i < 6:
                multiplier *= 1.2

            offspring += species.breed(multiplier)
        
        return offspring

    def prepare_generation(self):
        self.cleanup_agents()
        self.speciate()

    def evaluate_agents(self, inputs, outputs):
        for agent in self._agents:
            agent.error_sum = 0
            for input_, expected_outputs in zip(inputs, outputs):
                net_outputs, net_error = agent.activate_network(input_)
                for net_output, expected_output in zip(net_outputs, expected_outputs):
                    agent.error_sum += abs(expected_output - net_output)
                agent.error_sum += net_error

            agent.fitness = pow(max(0, 4 - agent.error_sum), 2)

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

