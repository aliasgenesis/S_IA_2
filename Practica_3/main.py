import csv
import numpy as np
import perceptron_with_error_graph as pwe

trainingPatternsFileName = "InputValues2.csv"
outputValuesFileName = "OutputValues2.csv"
epochs = 20

file = open(trainingPatternsFileName)
rows = len(file.readlines())
file.close()
#To obtain the number of rows from the CSV file

file = open(trainingPatternsFileName,'r')
reader = csv.reader(file,delimiter=',')
columns = len(next(reader))
file.close()
#To obtain the number of columns from the CSV file
      
net = pwe.neurona(columns, 0.1)
#Perceptron initialization.
                    
############################################################################                    
patterns = []

for i in range(columns):
    x = np.array(np.loadtxt(trainingPatternsFileName, delimiter=',', usecols=i))
    patterns.append(x)
X = np.array(patterns)

y = np.array(np.loadtxt(outputValuesFileName, delimiter=',', usecols=0))
#Obtaining training patterns in X and output values in y.
###########################################################################

net.train(X, y, epochs)

results = []
for i in range(rows):
    prediction = net.predict(X[:, i])
    results.append(prediction)
    
np.savetxt("Results.csv", results, delimiter=",", fmt='%.0f')