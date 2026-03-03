import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def MSE(y, pred):
    return np.mean((y - pred) ** 2) / 2

def MSEgrad(y,pred):
    return (pred-y)/y.size

losses = {MSE:MSEgrad}

def grad(y,pred,loss):
    return losses[loss](y,pred)
