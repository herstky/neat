import random

from kypy_neat.population import Population
from kypy_neat.phenotype import Phenotype
from kypy_neat.agent import Agent


class Experiment:
    def __init__(self):
        self.population = Population()
        self._best_performance = float('-inf')
        self._current_generation = 0
        self._num_generations = 500
        self.inputs = [[1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]]
        
        self.outputs = [[0], [1], [1], [0]]

    def shuffle_data(self, inputs, outputs):
        data = [x for x in zip(inputs, outputs)]
        random.shuffle(data)
        inputs = [x[0] for x in data]
        outputs = [x[1] for x in data]

        return inputs, outputs

    def evaluate_agents(self, inputs, outputs):
        inputs, outputs = self.shuffle_data(inputs, outputs)
        
        generation_champion = None
        for agent in self.population.agents:
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

            # NOTE should probably change this to fitness basis if 
            # adjusted_fitness calculation changes to prevent selecting a 
            # generation_champ that is elligible to be culled
            if not generation_champion:
                generation_champion = agent
            elif agent.error_sum < generation_champion.error_sum:
                generation_champion = agent 

        genotype_copy = generation_champion.genotype.generate_copy()
        phenotype = Phenotype(genotype_copy)
        self.population.generation_champion = Agent(phenotype)

    def record_species_results(self):
        for species in self.population.species:
            species.record_results()


    def print_generation_results(self):
        top_performance = float('-inf')
        print('******************************** GENERATION RESULTS ****************************************')
        print()
        for species in self.population.species:
            champion = species.champion
            if not champion:
                raise RuntimeError('No champion assigned')
            performance = max(0, (1 - champion.classification_error / 4) * 100)
            top_performance = max(top_performance, performance)
            age = species.age
            size = species.results['size']
            tot_shared_fitness = species.results['tot_shared_fitness']
            avg_shared_fitness = species.results['avg_shared_fitness']
            max_shared_fitness = species.results['max_shared_fitness']
            min_shared_fitness = species.results['min_shared_fitness']
            print(f'Species {species.species_id} -- '
                  f'Size: {size}, '
                  f'Total: {tot_shared_fitness:.2f}, '
                  f'Avg: {avg_shared_fitness:.2f}, '
                  f'Max: {max_shared_fitness:.2f}, '
                  f'Min: {min_shared_fitness:.2f}, '
                  f'Champ {champion.agent_id}: {performance:.1f}%')


        self._best_performance = max(self._best_performance, top_performance)

        print()
        print(f'Generation: {self._current_generation}, agents: {len(self.population.agents)}, species: {len(self.population.species)}')
        if top_performance > 75:
            end = ' !!!!!'
        else:
            end = ''
        print(f'Generation best: {top_performance:.1f}%{end}')
        print()
        print('*******************************************************************************************')
        print()

    def run(self):
        self.population.initialize_population(len(self.inputs[0]), 1)
        for generation in range(1, self._num_generations + 1):
            self._current_generation = generation
            self.population.prepare_generation()
            self.evaluate_agents(self.inputs, self.outputs)
            self.population.finish_generation()
            self.print_generation_results()
        
        print()
        print(f'Best overall performance: {self._best_performance:.5f}%')
        print()

class XOR(Experiment):
    pass