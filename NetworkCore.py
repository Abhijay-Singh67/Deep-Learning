import numpy as np
from helper import relu
class Linear:
    def __init__(self,input_features:int,output_features:int, layerID:int, activation=relu):
        self.__in = input_features
        self.__out = output_features
        self.__weights = np.random.randn(output_features,input_features)
        self.__bias = np.random.randn(output_features,1)
        self.id = layerID
        self.__activation = activation
    
    def forward(self,x):
        if(not isinstance(x,np.ndarray)):
            raise Exception(f"The input vector to layer {self.id} is not a numpy array!!")
        if(x.shape[0]!=self.__in):
            raise Exception(f"The input shape to layer {self.id} is incorrect!! Expected ({self.__in},{self.__out}) but {x.shape} was provided")
        try:
            out= (self.__weights@x)+self.__bias
        except:
            raise Exception(f"Error in propagating forward through the layer {self.id}")
        return self.__activation(out)
    
    def backward(self):
        #To be implemented
        pass

class Sequential:
    def __init__(self,*layers):
        self.__layers = list(layers)
    
    def forwardPass(self,x):
        out = x
        for i in self.__layers:
            out = i.forward(out)
        return out
        