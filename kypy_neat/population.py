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
        self._target_population = 150
        self._cull_fraction = 0.2 # fraction of species to be annihilated each generation
        self._species_floor = 8 # minimum number of species that must exist before any are annihilated
        self._breed_fraction = 1 # fraction of species that will breed each round
        self._min_breeding_species = 4 # minimum number of species that will breed each round
        self._generation_champion = None
        self._generation_champion_bonus_offspring = 0

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

    def ranked_species(self, descending):
        return sorted(self._species, key=lambda x: x.total_shared_fitness, reverse=descending)

    @property
    def aggregate_shared_fitness(self):
        return sum([species.total_shared_fitness for species in self._species])

    def get_species_of_agent(self, agent):
        for species in self._species:
            if species.in_species(agent):
                return species
        else:
            return None

    def create_species(self, representative):
        new_species = Species(representative)
        new_species.add(representative)
        self._species.append(new_species)
        return new_species

    def age_species(self):
        for species in self._species:
            species.age += 1

    def initialize_population(self, num_inputs, num_outputs):
        Genotype.initialize_minimal_topology(num_inputs, num_outputs)
        for _ in range(self._target_population):
            genotype = Genotype.base_genotype_factory()
            phenotype = Phenotype(genotype)
            self._agents.append(Agent(phenotype))

    def speciate(self):
        for agent in self._agents:
            for species in self._species:
                if species.compatible(agent):
                    species.add(agent)
                    break
            else:
                self.create_species(agent)

        Species.control_species_count()

    def kill_species_agents(self, species):
        agents = species.agents
        for agent in agents:
            agent.kill()

        self.cleanup_agents()

    def cull_species(self):
        for species in self._species:
            species.cull()

    def reset_species(self):
        for species in self._species:
            species.reset()

    def breed_species(self):
        aggregate_shared_fitness = self.aggregate_shared_fitness
        offspring = []
        champ = self._generation_champion

        champ_species = self.get_species_of_agent(champ)
        for _ in range(self._generation_champion_bonus_offspring):
            offspring.append(champ_species.generate_offspring(champ))

        num_remaining_offspring = self._target_population - len(offspring)
        for species in self._species:
            species_total_shared_fitness = species.total_shared_fitness
            species_offspring_share = num_remaining_offspring * species_total_shared_fitness / aggregate_shared_fitness

            offspring += species.breed(species_offspring_share)

        # while len(offspring) < self._target_population:
        #     offspring.append(champ_species.generate_offspring(champ))

        return offspring

    def prepare_generation(self):
        self._generation_champion = None
        self.speciate()
        self.remove_extinct_species() 

    def evaluate_agents(self, inputs, outputs):
        self._generation_champion = None
        for agent in self._agents:
            agent.error_sum = 0
            for input_, expected_outputs in zip(inputs, outputs):
                net_outputs, net_error = agent.activate_network(input_)
                for net_output, expected_output in zip(net_outputs, expected_outputs):
                    agent.error_sum += abs(expected_output - net_output)
                agent.error_sum += net_error

            agent.fitness = max(0, 4 - agent.error_sum)

            # NOTE should probably change this to fitness basis if 
            # adjusted_fitness calculation changes to prevent selecting a 
            # generation_champ that is elligible to be culled
            if not self._generation_champion:
                self._generation_champion = agent
            elif agent.error_sum < self._generation_champion.error_sum:
                self._generation_champion.champ = False
                self._generation_champion = agent 

    def record_species_results(self):
        for species in self._species:
            species.record_results()

    def finish_generation(self):
        self.record_species_results()
        self.cull_species()
        offspring = self.breed_species()
        self.reset_species()
        self.replace_agents(offspring)
        self.age_species()


    def replace_agents(self, offspring):
        self._agents = offspring

    def remove_extinct_species(self):
        species_copy = self._species[:]
        for species in species_copy:
            if not len(species.agents):
                species.annihilate()
                self._species.remove(species)