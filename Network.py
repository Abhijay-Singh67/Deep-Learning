from NetworkCore import Linear,Sequential
from helper import lin
import numpy as np
input = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
output = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(-1,1)
print(input.shape)
model = Sequential(
    Linear(1,1,lin)
)
model.fit(input,output,50,5)
print(model.predict(np.array([16]).reshape(-1,1)))
model.dump()