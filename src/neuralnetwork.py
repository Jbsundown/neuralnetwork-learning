import numpy as np 
import random

class NeuralNetwork:

	#layer_sizes: amount of neurons per layer, in the below example we have 2 input, two hiddeb hidden layers 
	#containing 3 neurons and then 5 neurons and 2 output neurons.
	# layer_sizes = (2,3,5,2)
	def __init__(self, layer_sizes):

		self.num_layers = len(layer_sizes)
		#This is a list containing the shapes of each weight matrix. The below has shapes from second to end and then beginning to penultimate.
		weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
		#A list that contains the weight matrices. Chosen randomly at the start of the program. If there are more inputs to a layer, the values are smaller.
		self.weights=[np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
		#A list that contains the biases.
		self.biases=[np.zeros((s,1)) for s in layer_sizes[1:]]

	#The output of the class. Self and a are the inputs to the network.
	def predict(self,a):
		#Gets the weights and biases to use.
		for w,b in zip(self.weights, self.biases):
			#Matrix multiplication plus the biases parsed through the activation function.
			#If this is the first itteration uses the inputs, all other loops will use a as the previous itteration's output.
			a = activationf(np.matmul(w,a) + b)
		return a

	"""Trains the network using a mini-batch stochastic gradient descent. training_data is a list of tuples that represent
	inputs and the desired outputs. Epochs is the number of epochs to train for and the batch size is the size of the 
	mini batches to use when sampling. eta is the learning rate. If test_data is given then the network evaluates itself after
	each epoch of training.
	In each epoch it starts by randomly shuffling the training data then partitions it into mini-batches for training.
	This is to sample randomly from the training data. Then for each mini_batch a single step of gradient descent is added."""
	def learn(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		n = len(training_data)
		if test_data: n_test = len(test_data)
		best_data = (0,0)

		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				getting_data = self.accuracy(test_data)
				print("Epoch {0}: out of {2}, {1}%".format(j, getting_data, n_test))
				if (getting_data > best_data[0]):
					best_data = (getting_data,j)
			else:
				print("Epoch {0} complete".format(j))
		if test_data:
			print("Best run was Epoch {0} with {1}%.".format(best_data[1],best_data[0]))

	#Update the network weights and biases by adding gradient descent using backpropagation to a single mini batch.
	#Computes the gradients for every training example in the mini_batch and then updates the weights and biases.
	def update_mini_batch(self, mini_batch, eta):
		#nabla is the gradient
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw
					for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
					for b, nb in zip(self.biases, nabla_b)]

	"""Backpropagation is a fast way of computing the gradient of the cost function.
		It returns a tuple (nabla_b, nabla_w) the represents the gradient for the cost function."""
	def backprop(self, x,y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		#list to store all the activations
		activations = [x] 
		#list to store all the z vectors
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = activationf(z)
			activations.append(activation)
		delta = self.cost_derivative(activations[-1], y) * \
			 sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		#l = 1 is the last layer of neurons, l= 2 is the penulatimate, etc.
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
		return(nabla_b,nabla_w)


	#Return the vector of partial derivatives
	def cost_derivative(self, output_activations, y):
		return (output_activations-y)

	def accuracy(self, test_data):
		test_results = [(np.argmax(self.predict(x)),y) for (x,y) in test_data]
		return sum(int(x==y) for (x,y) in test_results)/len(test_data)*100


#The activation function. A sigmoid function.
def activationf(x):
	return 1/(1+np.exp(-x))

	
def sigmoid_prime(z):
	return (activationf(z)*(1-activationf(z)))

