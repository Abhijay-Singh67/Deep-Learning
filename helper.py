import math as m
import numpy as np

def sigmoid(x):
    return 1/(1+m.exp(-x))

def relu(x):
    return max(0,x)

def vec_relu(x):
    return np.maximum(0,x)