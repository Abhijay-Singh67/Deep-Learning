from NetworkCore import *
layers = []
#This is where we generate the patterns
firstLayerInput = []
expectedOutput = []
for i in range(0,10):
    firstLayerInput.append(i)
    if i%2==0:
        expectedOutput.append(1)
    else:
        expectedOutput.append(0)


#Creating the instance of the layers
def Sequential(numOfLayers):
    for f in range(numOfLayers):
        x = int(input(f"The number of nuerons in {f} Layer: "))
        if f==numOfLayers-1:
            layers.append(Layers(x,f,outputLayer=True,inputLayer=False))
        elif f==0:
            layers.append(Layers(x,0,outputLayer=False,inputLayer=True))
        else:
            layers.append(Layers(x,f,outputLayer=False,inputLayer=False))   
    #initialising all the layers
    layers[0].Initialize(0)
    for g in range(1,len(layers)):
        layers[g].Initialize(layers[g-1].numberOfNuerons)


#Passing values to the layers
def passVals(firstLayerInput):
    for h in range(len(layers)):
        if h==0:
            layers[h].PassVals(firstLayerInput)
        else:
            layers[h].PassVals(layers[h-1].LayerOutput)

def CalculateLoss(Output):
    losses=[]
    for index in range(len(Output)):
        losses.append(binaryCrossEntropy(expectedOutput[index],Output[index]))
    loss = sum(losses)/len(losses)
    return loss

def Propogate(numOfepochs,firstLayerInput,layers):
    for epoch in range(numOfepochs):
        #doing forward propogation
        passVals(firstLayerInput)
        #doing backward propogation
        loss = CalculateLoss(layers[-1].LayerOutput)
        for i in range(len(layers)):
            for j in range(layers[i].numberOfNuerons):
                layers[i].Nuerons[j].Update(loss)
        acc = accuracy(layers[-1].LayerOutput,expectedOutput)
        ep = epoch
        print(f"Epoch{ep} ==>")
        print(f'Accuracy: {acc}')
        print(f'Loss: {loss}')

numOfLayers = int(input("Enter the number of layers you want: "))
Sequential(numOfLayers)
epochs = int(input("Enter the number of epochs you want: "))
Propogate(epochs,firstLayerInput,layers)
BestWeights = []
for i in range(numOfLayers):
    for j in range(layers[i].numberOfNuerons):
        BestWeights.append(layers[i].Nuerons[j].weights)
print(f"The weights are: {BestWeights}")
print(len(BestWeights))




    
    




