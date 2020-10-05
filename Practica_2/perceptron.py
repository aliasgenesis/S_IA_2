import numpy as np
import matplotlib.pyplot as plt
import csv

def graph(w_values):
    plt.clf()
    plt.scatter(X[0], X[1], color="black")
    plt.axhline(color="blue")
    plt.axvline(color="blue")
    x_values = [-3,3]
    y_values = [-(net.w[0][0]/net.w[1][0])*(-3) - (net.b / net.w[1][0]), 
                -(net.w[0][0]/net.w[1][0])*(3) - (net.b / net.w[1][0])]
    plt.plot(x_values, y_values, color="black")
    plt.pause(0.1)

class neurona:
    def __init__(self, dim, eta):   #Dimension y coeficiente de aprendizaje.
        self.n = dim
        self.eta = eta
        self.w = -1 + 2 * np.random.rand(dim, 1)  #x = min + (max - min)*rand()
        self.b = -1 + 2 * np.random.rand()

    def predict(self, x):
        y = np.dot(self.w.transpose(), x) + self.b
        if y >= 0:
            return 1
        else:
            return -1

    def train(self, X, y, epochs):  #x = matriz de entrenamiento, y = vector con resultado esperados, epochs = épocas.
        n, m = X.shape
        #n = 2. m = 4 en el ejemplo de la compuerta AND, OR y XOR.
        for i in range(epochs):
            for j in range(m):
                y_pred = self.predict(X[:, j])
                #what that line did is sliced the array, 
                #taking all rows (:) but keeping the column (j)
                if y_pred != y[j]:  #Si nuestro estimado es diferente a nuestro esperado, entrenamos.
                    self.w += self.eta*(y[j] - y_pred) * X[:, j].reshape(-1, 1)
                    self.b += self.eta*(y[j] - y_pred) 
                graph(self.w)

################################################################################################################

file = open("DataSet2.csv")
rows = len(file.readlines())
file.close()
#To obtain the number of rows from the CSV file

f = open("DataSet2.csv",'r')
reader = csv.reader(f,delimiter=',')
columns = len(next(reader))
f.close()
#To obtain the number of columns from the CSV file
      
net = neurona(columns-1, 0.1)
#Perceptron initialization.
                    
###########################################################################                    
patterns = []

for i in range(columns-1):
    x = np.array(np.loadtxt("DataSet2.csv", delimiter=',', usecols=i))
    patterns.append(x)
X = np.array(patterns)

y = np.array(np.loadtxt("DataSet2.csv", delimiter=',', usecols=columns-1))
#Obtaining training patterns in X and output values in y.

##########################################################################

net.train(X, y, 20)

for i in range(rows):
    print(net.predict(X[:, i]))
    
plt.clf()
plt.scatter(X[0], X[1], color="black")
plt.axhline(color="blue")
plt.axvline(color="blue")
x_values = [-3,3]
y_values = [-(net.w[0][0]/net.w[1][0])*(-3) - (net.b / net.w[1][0]), 
            -(net.w[0][0]/net.w[1][0])*(3) - (net.b / net.w[1][0])]
plt.plot(x_values, y_values, color="green")