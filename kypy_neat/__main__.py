# TODO: 
#       round down when reproducing and give extra offspring to champs
#       bias to generation champ
#       low performing agents are still able to perpetuate new species. culling doesn't affect low pop species
#       should use pareto dist to assign num offspring; 80% of outcomes are due to 20% of causes
#       its possible the necessary innovations require massive weight changes and so perform badly for a while

from kypy_neat.experiments import Experiment

def main():
    exp = Experiment()
    exp.run()

main()