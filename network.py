import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Input data
# 	number of samples = 3
# 	number of features per sample = 4

# X = [[1, 2, 3, 2.5],
# 	  [2.0, 5.0, -1.0, 2.0],
# 	  [-1.5, 2.7, 3.3, -0.8]]


nnfs.init()

# Returns a spiral shape dataset of 100 samples, with 3 classes
X, y = spiral_data(100, 3)

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


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(5, 2)

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)


