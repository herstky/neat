import time

from kypy_neat.species import Species
from kypy_neat.genotype import Genotype
from kypy_neat.phenotype import Phenotype
from kypy_neat.agent import Agent
from kypy_neat.utils.timer import timer


class Population:
    def __init__(self, size=150):
        self._agents = []
        self._agent_dict = {}
        self._species = []
        self._size = size
        self.generation_champion = None
        self._generation_champion_bonus_offspring = 1

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

    def get_agent(self, agent_id):
        if agent_id in self._agent_dict:
            return self._agent_dict[agent_id]
        
        return None

    def initialize_population(self, num_inputs, num_outputs):
        Genotype.initialize_minimal_topology(num_inputs, num_outputs)
        for _ in range(self._size):
            genotype = Genotype.base_genotype_factory()
            phenotype = Phenotype(genotype)
            agent = Agent(phenotype)
            self._agents.append(agent)
            self._agent_dict[agent.agent_id] = agent


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

        champ = self.generation_champion
        for _ in range(self._generation_champion_bonus_offspring):
            offspring.append(Species.generate_offspring(champ))

        num_remaining_offspring = self._size - len(offspring)
        for species in self._species:
            species_total_shared_fitness = species.total_shared_fitness
            species_offspring_share = num_remaining_offspring * species_total_shared_fitness / aggregate_shared_fitness

            offspring += species.breed(species_offspring_share)

        # while len(offspring) < self._target_population:
        #     offspring.append(champ_species.generate_offspring(champ))

        return offspring

    def prepare_generation(self):
        self.generation_champion = None
        self.speciate()
        self.remove_extinct_species() 

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
        self._agent_dict = {agent.agent_id: agent for agent in offspring}

    def remove_extinct_species(self):
        species_copy = self._species[:]
        for species in species_copy:
            if not len(species.agents):
                species.annihilate()
                self._species.remove(species)

    def set_generation_champion(self, champion):
        # Generation champion is copied in case the original gets culled
        genotype_copy = champion.genotype.generate_copy()
        phenotype = Phenotype(genotype_copy)
        agent_copy = Agent(phenotype) 
        agent_copy.fitness = champion.fitness
        self.generation_champion = agent_copy
