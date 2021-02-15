# TODO: improve selectivity
#       increase number of offspring relative to fitness
#       improve speciation and culling
#       Top priority: find root cause of top performance getting lost between generations
#       

from kypy_neat.experiments import Experiment

def main():
    exp = Experiment()
    exp.run()

main()