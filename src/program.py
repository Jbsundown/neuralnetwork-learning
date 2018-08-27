import neuralnetwork as nnw 
import numpy as np 
import mnist_loader as mnl  
"""
with np.load('../data/mnist.npz') as data:
	training_images = data['training_images']
	training_labels = data['training_labels']
	test_images = data['test_images']
	test_labels = data['test_labels']
	validation_images = data['validation_images']
	validation_labels = data['validation_labels']
"""

training_data, validation_data, test_data = mnl.load_data_wrapper()
layer_sizes = (784,30,10)

#print(len(training_images))
net = nnw.NeuralNetwork(layer_sizes)
#prediction = net.predict(training_images)
#print(np.argmax(prediction[0]))
net.learn(training_data, 30, 10, 3.0, test_data=test_data)
