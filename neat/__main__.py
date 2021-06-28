# NOTE champions are not getting copied

from kypy_neat.experiments import XOR, SinglePoleProblem

def evaluate_result(exp):
    agent = exp.load_agent('xor_agents/gen100_id15055')
    inputs = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    outputs = [[0], [1], [1], [0]]
    for _ in range(10):
        shuffled_inputs, shuffled_outputs = exp.shuffle_data(inputs, outputs)
        exp.evaluate_agent(agent, shuffled_inputs, shuffled_outputs)
        # print(f'Inputs: {shuffled_inputs}, Outputs: {shuffled_outputs}')
        print(f'Classification Error: {agent.classification_error}, '
            f'Actual Error: {agent.error_sum}, '
            f'Fitness: {agent.fitness}')

def main():
    exp = XOR()
    # evaluate_result(exp)
    # exp = SinglePoleProblem()
    exp.run()
    # exp.evaluate_solution()

main()