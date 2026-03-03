import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def siggrad(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relugrad(x):
    return ((x>=0).astype(float)+1e-10)

def lin(x):
    return x

def lingrad(x):
    return np.ones(x.shape)

def MSE(y, pred):
    return np.mean((y - pred) ** 2) / 2

def MSEgrad(y,pred):
    return (pred-y)/y.size


grads = {MSE:MSEgrad,relu:relugrad,sigmoid:siggrad,lin:lingrad}

def grad(y,pred,loss):
    return grads[loss](y,pred)

def actigrad(x,act):
    return grads[act](x)
