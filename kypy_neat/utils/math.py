import math


def sigmoid(x, coeff=1):
    return 1 / (1 + math.exp(-coeff * x))