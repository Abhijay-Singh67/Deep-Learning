from metrics import *
import random as r

class Nueron:
    def __init__(self,prevLayerLength,nueronIdentityNumber,output):
        self.weights = []
        self.identity = nueronIdentityNumber
        for a in range(prevLayerLength):
            self.weights.append(r.uniform(0,1))
        self.bias = r.uniform(0,1)
        self.isoutput = output
    
    #during forward propogation
    def calculate(self,prevLayerOutputs):
        output = 0
        for b in range(len(prevLayerOutputs)):
            output += prevLayerOutputs[b]*self.weights[b]
        output += self.bias
        if self.isoutput:
            output = sigmoid(output)
        else:
            output = relu(output)
        return output

    #during backward propogation   
    
    def Update(self,loss):
        for c in range(len(self.weights)):
            self.weights[c] = self.weights[c] - (0.01)*(derivative(loss,self.weights[c]))
        self.bias = self.bias - (0.01)*(derivative(loss,self.bias))


class Layers:
    def __init__(self,units,layerIdentityNumber,outputLayer,inputLayer):
        self.numberOfNuerons = units
        self.Nuerons = []
        self.identity = layerIdentityNumber
        self.isoutputLayer = outputLayer
        self.LayerOutput = []
        self.inputLayer = inputLayer

    #initialising the nuerons
    def Initialize(self,prevLayerLength):
        for d in range(self.numberOfNuerons):
            if self.inputLayer!=True:
                self.Nuerons.append(Nueron(prevLayerLength,f'{self.identity}{d}',self.isoutputLayer))
            else:
                self.Nuerons.append(Nueron(0,f'0{d}',output=False))
        print("The nuerons have been successfully initialised")

    #passing values to the nuerons

    def PassVals(self,previousLayerOutput):
        self.LayerOutput = []
        if self.inputLayer!=True:
            for e in range(self.numberOfNuerons):
                self.LayerOutput.append(self.Nuerons[e].calculate(previousLayerOutput))
        else:
            self.LayerOutput = previousLayerOutput
