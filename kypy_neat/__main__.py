from kypy_neat.experiments import XOR, SinglePoleProblem

def main():
    exp = SinglePoleProblem()
    # exp = XOR()
    exp.run()
    exp.evaluate_solution()

main()