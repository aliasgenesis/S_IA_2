import csv
import numpy as np
import ADALINE_Batch_Gradient_Descent as Adaline_BGD

trainingPatternsFileName = "InputValues2.csv"
outputValuesFileName = "OutputValues2.csv"
epochs = 40

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
    net = Adaline_BGD.adaline2(columns, 0.1)
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
individual_error = 0.0
results = []

for i in range(epochs):
    for j in range(number_of_neurons):
        net = neurons_array[j]
        individual_error += net.train(X, y[j], 1)
    individual_error /= number_of_neurons
    global_errors.append(individual_error)
    individual_error = 0.0
    Adaline_BGD.graphError(i, global_errors[i][0])


for i in range(number_of_neurons):
    net = neurons_array[i]
    individual_result = np.concatenate(net.predict(X).tolist())
    results.append(individual_result)
    
results = np.array(results).T
print(results)
print(results.shape)
np.savetxt("Results.csv", results, delimiter=",", fmt='%.5f')
    