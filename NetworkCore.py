import numpy as np
from helper import relu,MSE,grad,actigrad
class Linear:
    def __init__(self,input_features:int,output_features:int, activation=relu):
        self.__in = input_features
        self.__out = output_features
        self.__weights = np.random.randn(input_features,output_features)
        self.__bias = np.random.randn(1,output_features)
        self.id = 0
        self.activation = activation
        self.current_output=np.zeros(self.__bias.shape)
        self.current_activated_output=np.zeros(self.__bias.shape)
    
    def forward(self,x):
        if(not isinstance(x,np.ndarray)):
            raise Exception(f"The input vector to layer {self.id} is not a numpy array!!")
        if(x.shape[1]!=self.__in):
            raise Exception(f"The input shape to layer {self.id} is incorrect!! Expected ({self.__in},{self.__out}) but {x.shape} was provided")
        try:
            out= x@self.__weights+self.__bias
            self.current_output=out
        except:
            raise Exception(f"Error in propagating forward through the layer {self.id}")
        self.current_activated_output=self.activation(out)
        return self.current_activated_output
    
    def update(self,gradW,gradB,lr):
        self.__weights = self.__weights-lr*gradW
        self.__bias = self.__bias - lr*gradB
    
    def weights(self):
        return self.__weights

    def bias(self):
        return self.__bias

class Sequential:
    def __init__(self,*layers,loss=MSE,learning_rate=1e-3):
        self.__layers = list(layers)
        self.num_layers = len(layers)
        for i in range(len(layers)):
            self.__layers[i].id=i
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
                self.backProp(x_batch,y_batch,pred)
            predFull = self.forwardPass(x)
            print(f"Epoch {i+1}/{epochs} Training Loss:{self.__loss(y,predFull):.6f}")

    def backProp(self,x,y,pred):
        grads=[]
        delta=np.zeros(y.shape)
        for i in range(self.num_layers-1,-1,-1):
            layer = self.__layers[i]
            if(i==self.num_layers-1):
                delta = grad(y,pred,self.__loss)*(actigrad(layer.current_output,layer.activation))
            else:
                next_layer = self.__layers[i+1]
                delta = (delta @ next_layer.weights.T) * actigrad(layer.current_output, layer.activation)
            if i == 0:
                A_prev = x
            else:
                A_prev = self.__layers[i-1].current_activated_output
            gradW = (A_prev.T @ delta)
            gradB=(np.sum(delta,axis=0,keepdims=True))
            grads.append((layer,gradW,gradB))
        
        for layer,gradW,gradB in grads:
            layer.update(gradW,gradB,self.__lr)
    
    def predict(self,x):
        return self.forwardPass(x)
    
    def dump(self):
        with open("weights.txt","w") as file:
            text=""
            for i in self.__layers:
                text+=f"Layer: {i.id}\n"
                text+=str(i.weights().flatten())+"\n"
                text+=str(i.bias().flatten())+"\n"
            file.write(text)

