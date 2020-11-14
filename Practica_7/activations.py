import numpy as np

def linear(z, derivative = False):
	a = z
	if derivative:
		da = np.ones(z.shape)
		return a, da
	return a

def tanh(z, derivative = False):
	a = np.tanh(z)
	if derivative:
		da = (1 + a) * (1 - a)
		return a, da
	return a

def sigmoid(z, derivative = False):
	a = 1 / (1 + np.exp(-z))
	if derivative:
		da = a * (1 - a)
		return a, da
	return a

def relu(z, derivative = False):
	a = z * (z >= 0)
	if derivative:
		da = np.array(z >= 0 , dtype=float)
		return a, da
	return a

def activate(function_name):
	if function_name == 'linear':
		return linear
	elif function_name == 'tanh':
		return tanh
	elif function_name == 'sigmoid':
		return sigmoid
	elif function_name == 'relu':
		return relu
	else:
		raise ValueError('function_name unknown')