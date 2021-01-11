import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

class Layer_Dense:
	'''
		n_inputs = number of features (per data sample or per output of the previous layer)
		n_neurons = number of neurons (per layer) 
	'''
	def __init__(self, n_inputs, n_neurons):

		# Returns an array of shape n_inputs x n_neurons of the standard normal distribution
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 
		
		# Returns a zero array of shape 1 x n_neurons 
		self.biases = np.zeros((1, n_neurons))
		

	def forward(self, inputs):

		self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
	
	def forward(self, inputs):
		self.output = np.maximum(0,inputs)


class Activation_Softmax:

	def forward(self, inputs):
		# Exponentiation and subtraction of max (calculated per row)
		exp_values = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)

		# Normalization of the values
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		
		self.output = probabilities

# Returns a spiral shape dataset of 100 samples per class, with 3 classes
X, y = spiral_data(100, 3)

# 2 input features, 3 neurons
dense1 = Layer_Dense(2,3)

activation1 = Activation_ReLU()

# 3 input features, 3 output neurons (3 predicted classes)
dense2 = Layer_Dense(3,3)

activation2 = Activation_Softmax()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])