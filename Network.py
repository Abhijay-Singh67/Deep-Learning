from NetworkCore import Linear,Sequential
from helper import sigmoid
import numpy as np
input = np.array([[1,1,1],[2,2,2],[3,3,3]])
model = Sequential(
    Linear(3,5,0),
    Linear(5,1,1,sigmoid)
)
print(model.forwardPass(input).shape)
model.fit(input,np.array([0.05,0.065,0.076]),2)