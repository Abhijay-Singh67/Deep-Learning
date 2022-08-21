from NetworkCore import *

#Create input data here
modelInput = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
expectedOutput = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#Functionality
layers = []
X = int(input("Enter the number of layers you want: "))
layers.append(InputLayer(modelInput))
for g in range(1,X):
    unit = int(input(f"Enter the number of Nuerons in Layer {g}: "))
    act = input("Enter the activation function for this layer: ")
    layers.append(Layer(unit,g,layers[g-1].numOfNuerons,act))

lossFunction = input("Enter the loss function for your model: ")
epochs = int(input("Enter the number of epochs you want: "))

for h in range(epochs):
    print(f"Epoch {h} ==>")
    #forward propogation
    for i in range(1,len(layers)):
        layers[i].PassVals(layers[i-1].LayerOutput)
    
    #calculation of Loss
    losses = []
    for j in range(len(expectedOutput)):
        losses.append(chooseLoss(lossFunction,expectedOutput[j],layers[-1].LayerOutput[0]))
    loss = sum(losses)/len(losses)

    #calculating accuracy
    correct = 0
    for o in range(len(expectedOutput)):
        if expectedOutput[o]==layers[-1].LayerOutput[0]:
            correct += 1
    
    accuracy = correct/len(expectedOutput)

    #Backward propogation
    for k in range(1,len(layers)):
        layers[k].UpdateNuerons(loss)
    
    print(f"loss = {loss}")
    print(f"accuracy = {accuracy}")

weights = ''
for l in range(1,len(layers)):
    weights += str(l)
    weights += '\n'
    for m in layers[l].Nuerons:
        for n in m.weights:
            weights += f'{n},'
    weights += '\n'

weightFile = open("E:\Projects\Deep Learning\weights.txt","w")
weightFile.write(weights)
weightFile.close()


