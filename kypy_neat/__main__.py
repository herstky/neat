from kypy_neat.experiments import XOR, SinglePoleProblem

def main():
    exp = XOR()
    # exp = SinglePoleProblem()
    exp.run()
    # exp.evaluate_solution()

main()