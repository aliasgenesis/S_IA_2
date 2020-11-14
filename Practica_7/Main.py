import csv
import numpy as np
from MPL import *
import matplotlib.pyplot as plt
    
def main():
    plt.figure(1)
    
    print("----------Funciones----------")
    print("1) Seno")
    print("2) Coseno")   
    print("3) Tangente Hiperbolica")
    function = int(input("Opcion: "))
    
    trainingPatternsFileName = "InputValues.csv"
    x_funcFile = "X_Graph_FuncValues.csv"
    
    if function == 1:
        outputValuesFileName = "SinOutputValues.csv"
        y_funcFile = "Y_Graph_SinValues.csv"
        plt.title("SIN function", fontsize=20)
        
    elif function == 2:
        outputValuesFileName = "CosOutputValues.csv"
        y_funcFile = "Y_Graph_CosValues.csv"
        plt.title("COS function", fontsize=20)
    
    elif function == 3:
        outputValuesFileName = "TanhOutputValues.csv"
        y_funcFile = "Y_Graph_TanhValues.csv"
        plt.title("TANH function", fontsize=20)
        
    else:
        raise ValueError('Funcion Desconocida')
    # CSV documents selection.
    
    
    epochs = 10000
    learning_rate = 0.2
    entries = 1  # of columns for the trainingPatternsFileName.
    neurons_in_hidden_layer = 3
    output_layer_neurons = 1
        
    net = MLP((entries, neurons_in_hidden_layer, output_layer_neurons), ('tanh', 'linear'))
    # First parenthesis:
        # First position: Number of entrances X to the NN.
        # Second to n-1 position: Number of neurons in hidden layers.
        # Last position: Number of neurons in the output layer.
    # Second parenthesis: Activation functions for the hidden and output layers.
    
    ############################################################################                    
    X = []
    y = []
    
    x_func = []
    y_func = []
    
    X.append(np.array(np.loadtxt(trainingPatternsFileName, delimiter=',', usecols=0)))
    y.append(np.array(np.loadtxt(outputValuesFileName, delimiter=',', usecols=0)))
    #Obtaining training patterns in X and output values in y.

    x_func.append(np.array(np.loadtxt(x_funcFile, delimiter=',', usecols=0)))
    y_func.append(np.array(np.loadtxt(y_funcFile, delimiter=',', usecols=0)))
    #Obtaining graphing patterns in X and output values in y.

    ###########################################################################
    
    # Training section.

    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlim([-5, 5])
    plt.ylim([-5, 10])
        
    error_list = []
    
    for i in range(epochs):
        error, pred = net.train(X, y, 1, learning_rate)
        error_list.append(error)
        print("Epoch:", i, "Error:", error)

        if i%10 == 0:
            plt.clf()
            plt.scatter(x_func, y_func, s=40, c='#1f77b4')
            plt.plot(X[0], pred[0], color='red', linewidth=3)  
            plt.show()
            plt.pause(0.2)
            
            if error < 0.03:
                break
    
    plt.figure(2)
    plt.plot(error_list, color='red', linewidth=3)
        
    results = np.array(net.predict(X)).T
    np.savetxt("Results.csv", results, delimiter=",", fmt='%.4f')
    ###########################################################################
    
if __name__ == "__main__":
    main()