import math as m
def sigmoid(x):
    output = 1/(1+((1/m.e)**x))
    return output

def relu(x):
    output = max(0,x)
    return output


def binaryCrossEntropy(yTrue,yPred):
    if yTrue==0:
        loss = -1*(m.log(1-yPred + 0.0000001))
    elif yTrue==1:
        loss = -1*(m.log(yPred+0.0000001))
    return loss
    
def derivative(loss,weight):
    output = loss/weight
    return output

def accuracy(outputexpected,outputgiven):
    diff = []
    for z in range(len(outputexpected)):
        diff.append(m.fabs(outputexpected[z]-outputgiven[z]))
    accuracy = sum(diff)/len(diff)
    return accuracy