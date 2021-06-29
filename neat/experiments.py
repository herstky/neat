import random as rand
import math
import pickle

from neat.population import Population
from neat.genotype import Genotype
from neat.agent import Agent
from neat.utils.timer import timer


class Experiment:
    def __init__(self, num_generations, population_size):
        self._num_generations = num_generations
        self.population = Population(population_size)
        self._current_generation = 0

    def shuffle_data(self, inputs, outputs):
        data = [entry for entry in zip(inputs, outputs)]
        rand.shuffle(data)
        inputs = [input_ for input_, output in data]
        outputs = [output for input_, output in data]

        return inputs, outputs

    def record_species_results(self):
        for species in self.population.species:
            species.record_results()

    def epoch(self):
        raise NotImplementedError

    def print_generation_results(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def evaluate_agent(self, agent):
        raise NotImplementedError

    def initialize(self, num_inputs, num_outputs):
        Genotype.initialize(num_inputs, num_outputs)
        self.population.initialize()

    def save_agent(self, agent, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(agent, outfile)

    def load_agent(self, filename):
        with open(filename, 'rb') as infile:
            agent = pickle.load(infile)
        
        return agent

class XOR(Experiment):
    def __init__(self, num_generations=50, population_size=150):
        super().__init__(num_generations, population_size)
        self.inputs = [[1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]]
        self.outputs = [[0], [1], [1], [0]]
        self.initialize(len(self.inputs[0]), len(self.outputs[0]))
        self._results = {}

    def evaluate_agent(self, agent, inputs, outputs):
        agent.error_sum = 0
        agent.classification_error = 0
        for input_, expected_outputs in zip(inputs, outputs):
            net_outputs, net_error = agent.activate_network(input_)
            for net_output, expected_output in zip(net_outputs, expected_outputs):
                agent.error_sum += abs(expected_output - net_output)
                if net_output < 0.5 and expected_output == 1:
                    agent.classification_error += 1
                elif net_output >= 0.5 and expected_output == 0:
                    agent.classification_error += 1
            agent.error_sum += net_error

        agent.fitness = max(0, 4 - agent.error_sum)

    def reset_results(self):
        self._results = {}
        self._results['species'] = {}
        for species in self.population.species:
            self._results['species'][species.species_id] = {}

    def record_generation_results(self, generation_champion):
        self._results['gen'] = self._current_generation
        self._results['count'] = self.population.count
        self._results['species_count'] = self.population.species_count
        self._results['networks_evaluated'] = Agent.agents_created()

        self._results['gen_champ'] = {'id': generation_champion.agent_id,
                                      'fitness': generation_champion.fitness,
                                      'adjusted_fitness': generation_champion.adjusted_fitness,
                                      'hidden_nodes': len(generation_champion.phenotype.hidden_nodes),
                                      'connections': generation_champion.genotype.num_enabled_connection_genes,
                                      'error_sum': generation_champion.error_sum,
                                      'classification_error': generation_champion.classification_error} 

        for species in self.population.species:
            self._results['species'][species.species_id]['size'] = species.count
            self._results['species'][species.species_id]['total'] = species.total_shared_fitness
            self._results['species'][species.species_id]['max'] = species.max_shared_fitness
            self._results['species'][species.species_id]['min'] = species.min_shared_fitness
            self._results['species'][species.species_id]['avg'] = species.average_shared_fitness
            self._results['species'][species.species_id]['champ_id'] = species.champion.agent_id
            self._results['species'][species.species_id]['champ_error_sum'] = species.champion.error_sum

        if generation_champion.classification_error == 0:
            self.save_agent(generation_champion, f'xor_agents/gen{self._current_generation}_id{generation_champion.agent_id}.pkl')

    def epoch(self, inputs, outputs):
        self.reset_results()

        for agent in self.population.agents:
            inputs, outputs = self.shuffle_data(inputs, outputs)
            self.evaluate_agent(agent, inputs, outputs)

        generation_champion = min(self.population.agents, key=lambda agent: agent.error_sum)
        self.record_generation_results(generation_champion)
        self.population.set_generation_champion(generation_champion)

    def print_generation_results(self):
        print('******************************************* GENERATION RESULTS **************************************************')
        print()
        print(f'Generation {self._results["gen"]} -- '
              f'Agents: {self._results["count"]} | '
              f'Species: {self._results["species_count"]} | '
              f'Networks Evaluated: {self._results["networks_evaluated"]}')
        print()

        species_res = [{'id': species_id, 'res': res} for species_id, res in sorted(self._results['species'].items())]
        total_shared_fitness = 0
        for species in species_res:
            total_shared_fitness += species['res']['total']

        for species in species_res:
            print(f'Species {species["id"]} -- '
                  f'Size: {species["res"]["size"]} | '
                  f'Total: {species["res"]["total"]:.2f} | '
                  f'Avg: {species["res"]["avg"]:.2f} | '
                  f'Max: {species["res"]["max"]:.2f} | '
                  f'Min: {species["res"]["min"]:.2f} | '
                  f'Offspring Share: {100 * species["res"]["total"] / total_shared_fitness:.2f}% | '
                  f'Champion ID: {species["res"]["champ_id"]} | '
                  f'Champion Error Sum: {species["res"]["champ_error_sum"]:.2f}')

        champ_res = self._results['gen_champ']
        print()
        print(f'Champion {champ_res["id"]} -- '
              f'Error Sum: {champ_res["error_sum"]:.2f} | '
              f'Classification Error: {champ_res["classification_error"]} | '
              f'Hidden Nodes: {champ_res["hidden_nodes"]} | '
              f'Connections: {champ_res["connections"]}')
        print()
        print('****************************************************************************************************************')
        print()

    def run(self):
        for generation in range(1, self._num_generations + 1):
            self._current_generation = generation
            self.population.prepare_generation()
            self.epoch(self.inputs, self.outputs)
            self.print_generation_results()
            self.population.finish_generation()
