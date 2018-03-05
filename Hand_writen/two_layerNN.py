#import tensorflow as tf
# For this linear classification machine, I am not going to use tensorflow, import tensorflow just for testing purpose
from random import seed
from random import random
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import subprocess


# This is my first nueral network file, inspired by https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/





# This neural network is fully functional

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



# Initialize a network
# We have three layers, one input layers, one hidden layer and one output layer 
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = list()
	for i in range(n_hidden):
		weight_list = list()
		for j in range(n_inputs + 1):
			weight_list.append(0)
		hidden_layer.append({'weights':weight_list})
	#hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	# Here is an one line implementation
	output_layer = [{'weights':[0 for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network




def calculate_neuron_value(weights, inputs):
	bias = weights[-1]
	for i in range(len(inputs) -1):
		bias += weights[i]*inputs[i]
	return bias

# This machine uses sigmoid function as activation function
def activation_function(output):
	return 1.0/(1.0 + math.exp(-output))


# This will train a picture through the network

def forward_prop(network, row):
	inputs = row
	for layers in network:
		new_input = []
		for neurons in layers:
			neuron_value = calculate_neuron_value(neurons['weights'], inputs)
			neurons['output'] = activation_function(neuron_value)
			new_input.append(neurons['output'])
		inputs = new_input
	return inputs

def activation_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
# Very ineffecient 
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			# This is a fully connected network
			# Every neuron in the current layer will affect next layer in some kind, therefore, the lost is accumulated by all neurons in the next layer
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * activation_derivative(neuron['output'])



# This neural network uses error functions 
# The idea of an error function is to use to direct where the weight should go
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		for neuron in network[i]:
			for j in range(len(neuron['weights'])):
				neuron['weights'][j] += l_rate * neuron['delta']
			

if __name__ == '__main__':
	# Since we are going to make this network easy, the network will have only three inputs and one hidden layer
	# This line means that we will have 3 inputs and two neurons for the hidden layer. 
	network_created = initialize_network(3072, 100, 10)
	#for layer in network_created:
	#	print(layer)
	# This naive model will train over two patches of cifar 10
	batch1 = unpickle("cifar-10-batches-py/data_batch_1")
	batch2 = unpickle("cifar-10-batches-py/data_batch_2")
	print(type(batch1))
	#print(batch1['labels'])
	for key in batch1.keys():
		print(type(key))
	print(batch1[b'labels'][0])

	#print(batch1[b'filenames'])
	start = time.time()
	print(len(network_created))
	expected = list()
	#for rows in batch1[b'data']:
	for i in range(10):
		expected.append(0)
		if i == 5:
			expected.append(1)





	for i in range(5):
		outputs = forward_prop(network_created, batch1[b'data'][0])
		backward_propagate_error(network_created, expected)
		update_weights(network_created, batch1[b'data'][0], 0.2)
		print(outputs)



	end = time.time()
	print(((end - start)* 50000)/300)

	prediction = forward_prop(network_created, batch1[b'data'][0])

	if(prediction.index(max(prediction)) == 6):
		print("success rate for frog images:", max(prediction))
	#print(prediction)

	single_img_reshaped = np.transpose(np.reshape(batch1[b'data'][0],(3, 32,32)), (1,2,0))
	plt.imshow(single_img_reshaped)
	plt.show()


	#print(outputs)