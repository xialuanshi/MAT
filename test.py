import numpy as np
a = np.random.rand(3, 4, 1)
s1 =[0, 1, 2]
print(a)
b = a[:, [1, 2, 0]]
print(b)