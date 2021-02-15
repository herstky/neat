from kypy_neat.population import Population


class Experiment:
    def __init__(self):
        self.population = Population()
        self._best_performance = float('-inf')
        self._current_generation = 0
        self._num_generations = 100
        self.inputs = [[1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]]
        
        self.outputs = [[0], [1], [1], [0]]

    def evaluate_agents(self):
        for agent in self.population.agents:
            agent.error_sum = 0
            for input_, expected_output in zip(self.inputs, self.outputs):
                outputs, error = agent.activate_network(input_)
                output = outputs[0]
                agent.error_sum += abs(expected_output - output) + error

            agent.fitness = pow(max(0, 4 - agent.error_sum), 2)

    def print_generation_results(self):
        top_performance = float('-inf')
        print('******************************** GENERATION RESULTS ***********************************')
        print()
        for species in self.population.species:
            champion = species.champion
            performance = max(0, (1 - champion.error_sum / 4) * 100)
            top_performance = max(top_performance, performance)
            print(f'Species {species.species_id}, size: {len(species.agents)}, champion: {champion.agent_id}, performance: {performance:.4f}%')

        self._best_performance = max(self._best_performance, top_performance)

        print()
        print(f'Generation: {self._current_generation}, agents: {len(self.population.agents)}, species: {len(self.population.species)}')
        print(f'Generation best: {top_performance:.4f}%')
        print()
        print('**************************************************************************************')
        print()

    def run(self):
        self.population.initialize_population(len(self.inputs[0]), 1)
        for generation in range(1, self._num_generations + 1):
            self._current_generation = generation
            self.population.prepare_generation()
            self.population.evaluate_agents(self.inputs, self.outputs)
            self.population.finish_generation()
            self.print_generation_results()
        
        print()
        print(f'Best overall performance: {self._best_performance}')
        print()

class XOR(Experiment):
    pass