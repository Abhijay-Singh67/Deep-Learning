from NetworkCore import Linear,Sequential
from helper import sigmoid
import numpy as np

input = np.array([1,1,1]).reshape(-1,1)
model = Sequential(
    Linear(3,5,0),
    Linear(5,1,1,sigmoid)
)
print(model.forwardPass(input))