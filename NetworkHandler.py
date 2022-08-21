from NetworkCore import *
#Functionality
class Sequential:
    def __init__(self,lossFunction,numOfFeatures):
        self.layers = []
        self.features = numOfFeatures
        self.lossFunction = lossFunction

    def Compile(self,numOfLayers):
        for g in range(1,numOfLayers):
            unit = int(input(f"Enter the number of Nuerons in Layer {g}: "))
            act = input("Enter the activation function for this layer: ")
            self.layers.append(Layer(unit,g,self.features,act))    
    def Train(self,modelInput,expectedOutput,epochs):
        for h in range(epochs):
            print(f"Epoch {h} ==>")
            #forward propogation
            for i in range(1,len(self.layers)):
                if i==1:
                    for p in range(self.features):
                        self.layers[1].PassVals(modelInput[p])
                else:
                    self.layers[i].PassVals(self.layers[i-1].LayerOutput)
            
            #calculation of Loss
            losses = []
            for j in range(len(expectedOutput)):
                losses.append(chooseLoss(self.lossFunction,expectedOutput[j],self.layers[-1].LayerOutput[0]))
            loss = sum(losses)/len(losses)

            #calculating accuracy
            correct = 0
            for o in range(len(expectedOutput)):
                if expectedOutput[o]==self.layers[-1].LayerOutput[0]:
                    correct += 1
            
            accuracy = correct/len(expectedOutput)

            #Backward propogation
            for k in range(1,len(self.layers)):
                self.layers[k].UpdateNuerons(loss)
            
            print(f"loss = {loss}")
            print(f"accuracy = {accuracy}")   
    def export(self):
        weights = ''
        for l in range(1,len(self.layers)):
            weights += f'{l-1}-{l}\n'
            for m in self.layers[l].Nuerons:
                weights += f'{m.identity}\n'
                for n in m.weights:
                    weights += f'{n},'
            weights += '\n'
            weights+=f'{m.bias}'
            weights += '\n'
        weightFile = open("E:\Projects\Deep Learning\weights.txt","w")
        weightFile.write(weights)
        weightFile.close()






