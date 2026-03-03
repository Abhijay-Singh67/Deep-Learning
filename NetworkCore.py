import numpy as np
from helper import relu,MSE,MSEgrad
class Linear:
    def __init__(self,input_features:int,output_features:int, layerID:int, activation=relu):
        self.__in = input_features
        self.__out = output_features
        self.__weights = np.random.randn(input_features,output_features)
        self.__bias = np.random.randn(1,output_features)
        self.id = layerID
        self.__activation = activation
    
    def forward(self,x):
        if(not isinstance(x,np.ndarray)):
            raise Exception(f"The input vector to layer {self.id} is not a numpy array!!")
        if(x.shape[1]!=self.__in):
            raise Exception(f"The input shape to layer {self.id} is incorrect!! Expected ({self.__in},{self.__out}) but {x.shape} was provided")
        try:
            out= x@self.__weights+self.__bias
        except:
            raise Exception(f"Error in propagating forward through the layer {self.id}")
        return self.__activation(out)
    
    def update(self,gradW,gradB,lr):
        self.__weights = self.__weights-lr*gradW
        self.__bias = self.__bias - lr*gradB

class Sequential:
    def __init__(self,*layers,loss=MSE,learning_rate=1e-3):
        self.__layers = list(layers)
        self.__lr=learning_rate
        self.__loss=loss
    
    def forwardPass(self,x):
        out = x
        for i in self.__layers:
            out = i.forward(out)
        self.currentOutput = out
        return out

    def fit(self,x,y,epochs:int,batch_size=1):
        N = x.shape[0]
        for i in range(epochs):
            for j in range(0,N,batch_size):
                x_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]
                pred = self.forwardPass(x_batch)
                self.backProp(y_batch,pred)
            predFull = self.forwardPass(x)
            print(f"Epoch {i+1}/{epochs} Training Loss:{self.__loss(y,predFull)}")

    def backProp(self,y,pred):
        pass

        
        