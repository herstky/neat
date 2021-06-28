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

    def evaluate_agent(self, agent):
        raise NotImplementedError

    def save_agent(self, agent, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(agent, outfile)

    def load_agent(self, filename):
        with open(filename, 'rb') as infile:
            agent = pickle.load(infile)
        
        return agent


class XOR(Experiment):
    def __init__(self, num_generations=100, population_size=150):
        super().__init__(num_generations, population_size)
        self.inputs = [[1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]]
        
        self.outputs = [[0], [1], [1], [0]]
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
        self.population.initialize_population(len(self.inputs[0]), 1)
        for generation in range(1, self._num_generations + 1):
            self._current_generation = generation
            self.population.prepare_generation()
            self.epoch(self.inputs, self.outputs)
            self.print_generation_results()
            self.population.finish_generation()


class SinglePoleProblem(Experiment):
    def __init__(self, max_steps=100000, num_tests=10, num_generations=7, population_size=150):
        super().__init__(num_generations, population_size)
        self._max_steps = max_steps
        self.num_tests = num_tests

    def evaluate_agent(self, agent):
        return self.evaluate_agent_helper(agent) / self._max_steps * 100
        
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
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        force = force_mag if action > 0 else -force_mag
        
        temp = (force + pole_mass_length * theta_dot * theta_dot * sin_theta) / total_mass
        theta_acc = (gravity * sin_theta - cos_theta * temp) / (length * (4 / 3 - pole_mass * cos_theta * cos_theta / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

        x += tau * x_dot
        x_dot += tau * x_acc
        theta += tau * theta_dot
        theta_dot += tau * theta_acc

        return (x, x_dot, theta, theta_dot)

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
            self.print_generation_results()
            self.population.finish_generation()

    def evaluate_solution(self):
        with open('spp_solution.obj', 'rb') as infile:
            agent = pickle.load(infile)

        for i in range(1, self.num_tests + 1):
            performance = self.evaluate_agent(agent)

            print(f'Test: {i} -- Performance: {performance:.2f}%')
