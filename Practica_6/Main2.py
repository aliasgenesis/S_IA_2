import csv
import numpy as np
from MPL import *
import matplotlib.pyplot as plt

# Error graphing function.
def graphError(x_coordinate, y_coordinate):    
    plt.scatter(x_coordinate, y_coordinate)
    plt.plot(x_coordinate, y_coordinate, 'ro', markersize=14)
    plt.pause(0.0001)
    
def main():
    logic_gate = input("Compuerta logica a trabajar (xor/xnor): ")
    
    trainingPatternsFileName = "InputValues1.csv"
    
    if logic_gate == "xor":
        outputValuesFileName = "OutputValues1.csv"
    elif logic_gate == "xnor":
        outputValuesFileName = "OutputValues2.csv"
    else:
        raise ValueError('Compuerta Desconocida')
    # CSV documents selection.
    
    epochs = 10000
    learning_rate = 0.5
    neurons_in_hidden_layer = 2
    
    file = open(trainingPatternsFileName)
    rows = len(file.readlines())
    file.close()
    #To obtain the number of rows from the CSV file
    
    file = open(trainingPatternsFileName,'r')
    reader = csv.reader(file,delimiter=',')
    entries = len(next(reader))
    file.close()
    #To obtain the number of columns from the CSV file
    
    file = open(outputValuesFileName,'r')
    reader = csv.reader(file,delimiter=',')
    output_layer_neurons = len(next(reader))
    file.close()
    #To obtain the number of neurons for the program. The number of output columns tells us the number of neurons.
    
    net = MLP((entries, neurons_in_hidden_layer, output_layer_neurons), ('tanh', 'sigmoid'))
    # First parenthesis:
        # First position: Number of entrances X to the NN.
        # Second to n-1 position: Number of neurons in hidden layers.
        # Last position: Number of neurons in the output layer.
    # Second parenthesis: Activation functions for the hidden and output layers.
    
    ############################################################################                    
    patterns = []
    y = []
    
    for i in range(entries):
        x = np.array(np.loadtxt(trainingPatternsFileName, delimiter=',', usecols=i))
        patterns.append(x)
    X = np.array(patterns)
    
    for i in range(output_layer_neurons):
        y.append(np.array(np.loadtxt(outputValuesFileName, delimiter=',', usecols=i)))
    #Obtaining training patterns in X and output values in y.
    ###########################################################################
    
    # Training section.
    
    plt.figure(1)
    
    for i in range(epochs):
        error = net.train(X, y, 1, learning_rate)
        graphError(i, error)
        if error < 0.075:
            break
    
    ###########################################################################
    
    #Dibujar superficie de decisión
    plt.figure(2)
    
    if logic_gate == "xor":
        plt.title("XOR with MLP")
        plt.plot(0,0,'r*')
        plt.plot(0,1,'b*')
        plt.plot(1,0,'b*')
        plt.plot(1,1,'r*')
    elif logic_gate == "xnor":
        plt.title("XNOR with MLP")
        plt.plot(0,0,'b*')
        plt.plot(0,1,'r*')
        plt.plot(1,0,'r*')
        plt.plot(1,1,'b*')
    
    xx, yy = np.meshgrid(np.arange(-1, 2.1, 0.1), np.arange(-1, 2.1, 0.1))
    x_input = [xx.ravel(), yy.ravel()]
    zz = net.predict(x_input)
    zz = zz.reshape(xx.shape)
    
    plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.RdBu)
    
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.grid()
    plt.show()
    
    results = np.array(net.predict(X)).T
    np.savetxt("Results.csv", results, delimiter=",", fmt='%.0f')
    
if __name__ == "__main__":
    main()