from kypy_neat.population import Population


class Experiment:
    def __init__(self):
        self.population = Population()
        self._best_performance = float('-inf')
        self._current_generation = 0
        self._num_generations = 1000
        self.inputs = [[1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]]
        
        self.outputs = [[0], [1], [1], [0]]

    def print_generation_results(self):
        top_performance = float('-inf')
        print('******************************** GENERATION RESULTS ***********************************')
        print()
        for species in self.population.species:
            champion = species.champion
            if not champion:
                raise RuntimeError('No champ')
            performance = max(0, (1 - champion.error_sum / 4) * 100)
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
        print(f'Best overall performance: {self._best_performance:.5f}%')
        print()

class XOR(Experiment):
    pass