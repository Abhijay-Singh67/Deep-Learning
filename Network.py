from NetworkCore import Linear
import numpy as np

input = np.array([1,1,1]).reshape(-1,1)
layer = Linear(3,5,0)
out = layer.forward(input)
print(out)