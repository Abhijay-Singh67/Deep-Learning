import random as r
from metrics import *
#Base clases
class Nueron:
    def __init__(self,prevLayerLength,activationFunction,identityNumber):
        self.bias = r.uniform(0,1)
        self.weights = []
        for a in range(prevLayerLength):
            self.weights.append(r.uniform(0,1))
        self.activationFunction = activationFunction
        self.identity = identityNumber
    
    #forward propogation
    def calculate(self,prevLayerOutput):
        output = 0
        for b in range(len(self.weights)):
            for p in range(len(prevLayerOutput)):
                output += self.weights[b]*prevLayerOutput[p]
        output += self.bias
        output = chooseActivation(self.activationFunction,output)
        return output
    
    #backward propogation

    def updateNueron(self,loss):
        for c in range(len(self.weights)):
            self.weights[c] = gradientDescent(self.weights[c],loss,0.01)
            self.bias = gradientDescent(self.bias,loss,0.01)

class Layer:
    def __init__(self,units,identityNumber,prevLayerLength,activation):
        self.identity = identityNumber
        self.numOfNuerons = units
        self.Nuerons = []
        self.activation = activation
        self.LayerOutput = []

        #initialising the nuerons
        for d in range(units):
            self.Nuerons.append(Nueron(prevLayerLength,self.activation,f'{self.identity}d'))

    def PassVals(self,prevLayerOutput):
        if self.identity==1:
            for e in self.Nuerons:
                self.LayerOutput.append(e.calculate(prevLayerOutput))
        else:
            self.LayerOutput = []
            for e in self.Nuerons:
                self.LayerOutput.append(e.calculate(prevLayerOutput))

    def UpdateNuerons(self,loss):
        for f in self.Nuerons:
            f.updateNueron(loss)