##All the metrics used are derived from this file
import math as m
#Activation functions

def sigmoid(x):
    output = 1/(1+(1/m.e)**x)
    return output

def tanh(x):
    t1 = (m.e)**x
    t2 = (1/m.e)**x
    output = (t1 - t2)/(t1+t2)
    return output

def relu(x):
    output = max(0,x)
    return output

def leakyrelu(x):
    output = max(0.01*x,x)
    return output

#Loss functions

#For regression
def MSE(yTrue,yPred):
    loss = (yTrue-yPred)**2
    return loss/2

def MAE(yTrue,yPred):
    loss = m.fabs(yTrue-yPred)/2
    return loss

def huberLoss(yTrue,yPred,delta):
    if m.fabs(yTrue-yPred)<=delta:
        loss = ((yTrue-yPred)**2)/2
    else:
        loss = delta*m.fabs(yTrue-yPred) - (delta**2)/2
    
    return loss

#for classification

def binaryCrossEntropy(yTrue,yPred):
    if yTrue==0:
        loss = -1*(m.log(1-yPred))
    elif yTrue==1:
        loss = -1*(m.log(yPred))
    return loss

#Optimizers

def gradientDescent(weight,loss,learningRate):
    weight = weight - learningRate*(derivative(loss,weight))
    return weight

#side functions
def derivative(x,y):
    return x/y
def chooseActivation(name,x):
    if name=='sigmoid':
        return sigmoid(x)
    elif name=='tanh':
        return tanh(x)
    elif name=='relu':
        return relu(x)
    elif name=='leakyrelu':
        return leakyrelu(x)
    else:
        raise(RuntimeError("The specified activation function doesn't exist!!"))

def chooseLoss(name,yTrue,yPred):
    if name=='MSE':
        return MSE(yTrue,yPred)
    elif name=="MAE":
        return MAE(yTrue,yPred)
    elif name=='huberloss':
        return huberLoss(yTrue,yPred)
    elif name=="binaryCrossEntropy":
        return binaryCrossEntropy(yTrue,yPred)
    else:
        raise(RuntimeError("The specified Loss function doesn't exist!!"))
