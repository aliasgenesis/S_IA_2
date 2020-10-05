import numpy as np
import matplotlib.pyplot as plt

x_axis = []
y_axis = []

plt.ylabel('Error', fontsize=30)
plt.xlabel('Epocas', fontsize=30)
plt.title('Graficacion del error del perceptron', fontsize=40)

def graphError(x_coordinate, y_coordinate):
    plt.scatter(x_coordinate, y_coordinate)
    plt.plot(x_coordinate, y_coordinate, '*', markersize=14)
    plt.pause(0.5)
    
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
        average = 0
        for i in range(epochs):
            for j in range(m):
                y_pred = self.predict(X[:, j])
                #what that line did is sliced the array, 
                #taking all rows (:) but keeping the column (j)
                if y_pred != y[j]:  #Si nuestro estimado es diferente a nuestro esperado, entrenamos.
                    self.w += self.eta*(y[j] - y_pred) * X[:, j].reshape(-1, 1)
                    self.b += self.eta*(y[j] - y_pred) 
                average += (y[j] - y_pred)**2
            average /= m
            graphError(i, average)
            x_axis.append(i)
            y_axis.append(average)
            average = 0
        
        plt.plot(y_axis, 'c')