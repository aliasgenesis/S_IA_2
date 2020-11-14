import numpy as np 
from activations import *
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layers_dim, activations):
        #Atributes
        self.W = [None]	#Matrix with sinaptic weights of each layer. As the first layer (the entries) doesn't have weights, we use the None to make sure the first space in the matrix has nothing, but the rest does.
        self.b = [None]	#The same as W, but with the Bias.
        self.f = [None] #The same as W, but here we will gather every activation function for each individual layer.
        self.n = layers_dim
        self.L = len(layers_dim) - 1 #Max number of layers.
        
        #Initialization of sinaptic weights and bias.
        for l in range(1, self.L + 1):
            self.W.append(-1 + 2 * np.random.rand(self.n[l], self.n[l-1]))
            self.b.append(-1 + 2 * np.random.rand(self.n[l], 1))
            
        #Fill activation functions list
        for act in activations:
            self.f.append(activate(act))
            
    def predict(self, X):
        a = np.asanyarray(X)
        for l in range(1, self.L + 1):
            z = np.dot(self.W[l], a) + self.b[l]
            a = self.f[l](z)
        return a
    
    # def train(self, X, Y, epochs=1000, learning_rate=0.2):
    def train(self, X, Y, epochs, learning_rate):
        X = np.asanyarray(X)
        Y = np.asanyarray(Y).reshape(self.n[-1], -1)
        P = X.shape[1]
        error = 0
        
        for _ in range(epochs):
            #Stocastic Gradient Descend
            for p in range(P):
                A = [None] * (self.L + 1)
                dA = [None] * (self.L + 1)
                lg = [None] * (self.L + 1)
                
                #Propagation
                A[0] = X[:,p].reshape(self.n[0], 1)
                for l in range(1, self.L + 1):
                    z = np.dot(self.W[l], A[l-1]) + self.b[l]
                    A[l], dA[l] = self.f[l](z, derivative=True)
                    
                #Backpropagation
                for l in range(self.L, 0, -1):
                    if l == self.L:
                        #lg = Local Gradient
                        lg[l] = (Y[:, p] - A[l]) * dA[l]
                    else:
                        lg[l] = np.dot(self.W[l+1].T, lg[l+1]) * dA[l]
                
                #Weights updates
                for l in range(1, self.L + 1):
                    self.W[l] += learning_rate * np.dot(lg[l], A[l-1].T)
                    self.b[l] += learning_rate * lg[l]
                    
        predictions = self.predict(X)
        for i in range(len(predictions[0])):
            # error += (Y[0][i] - predictions[0][i])
            error += abs((Y[0][i] - predictions[0][i]))
        # print(error)
        error /= len(Y[0])
        return error, predictions
                