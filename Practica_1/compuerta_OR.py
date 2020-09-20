import numpy as np    

def predict(x, w, b):
    y = np.dot(w.transpose(), x) + b
    print(y)
    
    if y >= 0:
        return 1
    else:
        return -1
    
x_Vector = np.array([[1, 1, -1, -1], [1, -1, 1, -1]])
compuerta_OR = np.array([1, 1, 1, -1])

weights = np.array([0.5, 0.5])
bias = 0.5

for i in range(4):
    print(predict(x_Vector[:,i], weights, bias))