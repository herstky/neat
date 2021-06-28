import math

def sigmoid(x, coeff=1, offset=0):
    return 1 / (1 + math.exp(-coeff * (x + offset)))