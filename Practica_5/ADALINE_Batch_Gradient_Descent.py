import numpy as np
import matplotlib.pyplot as plt

plt.ylabel('Error', fontsize=30)
plt.xlabel('Epocas', fontsize=30)
plt.title('Graficacion del error del ADALINE con BGD', fontsize=40)

def graphError(x_coordinate, y_coordinate):
    plt.scatter(x_coordinate, y_coordinate)
    plt.plot(x_coordinate, y_coordinate, 'ro', markersize=14)
    plt.pause(0.3)

class adaline2:
    def __init__(self, dimensions, learning_rate):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.w_vector = -1 + 2 * np.random.rand(dimensions, 1)
        self.b_value  = -1 + 2 * np.random.rand()
        
    def predict(self, x_vector):
        return np.dot(self.w_vector.transpose(), x_vector) + self.b_value
    
    def train(self, X_matrix, y_vector, epochs):
        n, m = X_matrix.shape
        error = 0.0
        for i in range(epochs):
            w_sumatory_vector = np.zeros((self.dimensions, 1))
            b_sumatory = 0
            
            for j in range(m):
                y_estimated = self.predict(X_matrix[:,j])
                error += y_vector[j] - y_estimated
                w_sumatory_vector += (y_vector[j] - y_estimated) * X_matrix[:,j].reshape(-1, 1)
                b_sumatory        += (y_vector[j] - y_estimated)
            self.w_vector += (self.learning_rate / m) * w_sumatory_vector
            self.b_value  += (self.learning_rate / m) * b_sumatory
            return error