import csv
import numpy as np
import perceptron_with_error_graph as pwe

trainingPatternsFileName = "InputValues1.csv"
outputValuesFileName = "OutputValues1.csv"
epochs = 10

file = open(trainingPatternsFileName)
rows = len(file.readlines())
file.close()
#To obtain the number of rows from the CSV file

file = open(trainingPatternsFileName,'r')
reader = csv.reader(file,delimiter=',')
columns = len(next(reader))
file.close()
#To obtain the number of columns from the CSV file

file = open(outputValuesFileName,'r')
reader = csv.reader(file,delimiter=',')
number_of_neurons = len(next(reader))
file.close()
#To obtain the number of neurons for the program. The number of output columns tells us the number of neurons.

neurons_array = []
      
for i in range(number_of_neurons):
    net = pwe.neurona(columns, 0.1)
    neurons_array.append(net)
#Perceptron initialization.
                    
############################################################################                    
patterns = []
y = []

for i in range(columns):
    x = np.array(np.loadtxt(trainingPatternsFileName, delimiter=',', usecols=i))
    patterns.append(x)
X = np.array(patterns)

for i in range(number_of_neurons):
    y.append(np.array(np.loadtxt(outputValuesFileName, delimiter=',', usecols=i)))
#Obtaining training patterns in X and output values in y.
###########################################################################

global_errors = []
for i in range(number_of_neurons):
    net.train(X, y[i], epochs)

    individual_error = []
    for i in range(rows):
        prediction = net.predict(X[:, i])
        individual_error.append(prediction)
    global_errors.append(individual_error)

global_errors = np.array(global_errors).T
    
np.savetxt("Results.csv", global_errors, delimiter=",", fmt='%.0f')