import random as rand
import math
import pickle

from multiprocessing import Pool, Array, Value, Queue

from kypy_neat.population import Population
from kypy_neat.agent import Agent
from kypy_neat.utils.timer import timer


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


class XOR(Experiment):
    def __init__(self, num_generations=100, population_size=150):
        super().__init__(num_generations, population_size)
        self.inputs = [[1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]]
        
        self.outputs = [[0], [1], [1], [0]]

    def epoch(self, inputs, outputs):
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

            if not generation_champion:
                generation_champion = agent
            elif agent.error_sum < generation_champion.error_sum:
                generation_champion = agent 

        self.population.set_generation_champion(generation_champion)

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

        print()
        print(f'Generation: {self._current_generation} -- Agents: {len(self.population.agents)}, Species: {len(self.population.species)}, Networks Evaluated: {Agent.agents_created()}')
        print(f'Generation Best: {top_performance:.1f}%, Hidden Nodes: {len(self.population.generation_champion.phenotype.hidden_nodes)}')
        print()
        print('*******************************************************************************************')
        print()

    def run(self):
        self.population.initialize_population(len(self.inputs[0]), 1)
        for generation in range(1, self._num_generations + 1):
            self._current_generation = generation
            self.population.prepare_generation()
            self.epoch(self.inputs, self.outputs)
            self.population.finish_generation()
            self.print_generation_results()

class SinglePoleProblem(Experiment):
    def __init__(self, max_steps=100, num_tests=5, num_generations=20, population_size=10):
        super().__init__(num_generations, population_size)
        self._max_steps = max_steps
        self.num_tests = num_tests

    def evaluate_agent(self, agent):
        total = 0
        for i in range(self.num_tests):
            total += self.evaluate_agent_helper(agent) / self._max_steps * 100
        
        return total / self.num_tests

    def evaluate_agent_helper(self, agent):
        one_degree = 2 * math.pi / 360
        twelve_degrees = 12 * one_degree
        twenty_four_degrees = 24 * one_degree

        x = rand.uniform(-2.4, 2.4)
        x_dot = rand.uniform(-1, 1)
        theta = rand.uniform(-0.2, 0.2)
        theta_dot = rand.uniform(-1.5, 1.5)

        action = 0

        step = 0
        while (step < self._max_steps):
            bias = 1
            x_n = (x + 2.4) / 4.8
            x_dot_n = (x_dot + 0.75) / 1.5
            theta_n = (theta + twelve_degrees) / twenty_four_degrees
            theta_dot_n = (theta_dot + 1.0) / 2.0
            inputs = (bias, x_n, x_dot_n, theta_n, theta_dot_n)
            outputs, error = agent.activate_network(inputs)
            if error:
                return step

            out1, out2 = outputs
            if out1 > out2:
                action = 0
            else:
                action = 1
            
            x, x_dot, theta, theta_dot = self.move_cart(action, x, x_dot, theta, theta_dot)

            if (x < -2.4 or x > 2.4 or theta < -twelve_degrees or theta > twelve_degrees):
                return step

            step += 1
        
        return step

    def move_cart(self, action, x, x_dot, theta, theta_dot):
        gravity = 9.8
        cart_mass = 1.0
        pole_mass = 0.1
        total_mass = cart_mass + pole_mass
        length = 0.5
        pole_mass_length = pole_mass * length
        force_mag = 10.0
        tau = 0.02

        force = force_mag if action > 0 else -force_mag
        
        temp = (force + pole_mass_length * theta_dot * theta_dot * math.sin(theta)) / total_mass
        theta_acc = (gravity * math.sin(theta) - math.cos(theta) * temp) / (length * (4 / 3 - pole_mass * math.cos(theta) * math.cos(theta) / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * math.cos(theta) / total_mass

        x += tau * x_dot
        x_dot += tau * x_acc
        theta += tau * theta_dot
        theta_dot += tau * theta_acc

        return (x, x_dot, theta, theta_dot)

    # @timer
    # def epoch(self):
    #     generation_champion = None
    #     for agent in self.population.agents:
    #         agent.fitness = self.evaluate_agent(agent)

    #         if not generation_champion:
    #             generation_champion = agent
    #         elif agent.fitness > generation_champion.fitness:
    #             generation_champion = agent 

    #     self.population.set_generation_champion(generation_champion)
    #     with open('spp_solution.obj', 'wb') as outfile:
    #         pickle.dump(generation_champion, outfile)

    def process_agent(self, agent):
        return (agent.agent_id, self.evaluate_agent(agent))

    @timer
    def epoch(self):
        with Pool() as pool:
            output = pool.map(self.process_agent, self.population.agents)

        for agent_id, fitness in output:
            agent = self.population.get_agent(agent_id)
            agent.fitness = fitness

        generation_champion = max(self.population.agents, key=lambda agent: agent.fitness)
        self.population.set_generation_champion(generation_champion)
        with open('spp_solution.obj', 'wb') as outfile:
            pickle.dump(generation_champion, outfile)

        
    def print_generation_results(self):
        print(f'Generation: {self._current_generation}, '
              f'Best: {self.population.generation_champion.fitness:.2f}%, '
              f'Hidden Nodes: {len(self.population.generation_champion.phenotype.hidden_nodes)}, '
              f'Networks Evaluated: {Agent.agents_created()}')

    def run(self):
        self.population.initialize_population(5, 2)
        for generation in range(1, self._num_generations + 1):
            self._current_generation = generation
            self.population.prepare_generation()
            self.epoch()
            self.population.finish_generation()
            self.print_generation_results()

    def evaluate_solution(self):
        with open('spp_solution.obj', 'rb') as infile:
            agent = pickle.load(infile)

        for i in range(1, self.num_tests + 1):
            performance = self.evaluate_agent(agent)

            print(f'Test: {i} -- Performance: {performance:.2f}%')
