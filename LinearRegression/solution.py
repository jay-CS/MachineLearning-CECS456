import numpy as np 
import matplotlib.pyplot as plt
from helper import *

'''
Homework1: perceptron classifier
'''
def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data):
	'''
	This function is used for plot image and save it.

	Args:
	data: Two images from train data with shape (2, 16, 16). The shape represents total 2
	      images and each image has size 16 by 16. 

	Returns:
		Do not return any arguments, just save the images you plot for your report.
	'''
	##
	row = 0
	col = 0 
	plt.xlim([0,18])
	plt.ylim([0,18])
	d = np.transpose(data[0])
	for x in d:
		row = row + 1
		for y in x:
			if(round(y) == 1.0):
				plt.plot(col, row, marker = "o",color = "Black")
			else:
				plt.plot(col, row, marker = "x", color = "blue")
			col = col + 1
		col = 0
	plt.show()


	plt.xlim([1,18])
	plt.ylim([1,18])
	row = 0
	col = 0 
	for x in data[1]:
		row = row + 1
		for y in x:
			col = col + 1
			if(round(y) == 1.0):
				plt.plot(col,row,marker = "o", color = "Black")
			else:
				plt.plot(col,row,marker = "x",  color = "blue")
		col = 0

	
	plt.show()
	plt.imshow(data[0])
	plt.show()
	plt.imshow(data[1])
	plt.show()
	





def show_features(data, label):
	'''
	This function is used for plot a 2-D scatter plot of the features and save it. 

	Args:
	data: train features with shape (1561, 2). The shape represents total 1561 samples and 
	      each sample has 2 features.
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
	'''
	plt.xlim(-1.0,0)
	plt.ylim(-1.0,.1)
	plt.xlabel("Symmetry")
	plt.ylabel("Average Intensity")
	cont = 0
	# Took absolute value in order to plot actual numbers
	for i in data: 
		if(label[cont] == 1.0):
			plt.plot(i[0],i[1], marker = "*", color = "red")
		else:
			plt.plot(i[0],i[1], marker = "+",  color = "blue" )
		cont = cont + 1

	plt.show()



def perceptron(data, label, max_iter, learning_rate):
	'''
	The perceptron classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
	'''
	#intial weights
	# iniciate the weights and bias
	w = np.zeros((1,3))
	#w = np.array([0.3,         1.46403906, -0.17050937])
	cont = 0
	while(cont < max_iter):
		for xi, target in zip(data, label):
			if(sign(np.dot(w,xi)) != target):
				w += learning_rate * (xi * target) 
				cont += 1
	return w






def show_result(data, label, w):
	'''
	This function is used for plot the test data with the separators and save it.
	
	Args:
	data: test features with shape (424, 2). The shape represents total 424 samples and 
	      each sample has 2 features.
	label: test data's label with shape (424,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the image you plot for your report.
	'''
	plt.xlim(-1.0,.1)
	plt.ylim(-1.0,.2)
	plt.xlabel("Symmetry")
	plt.ylabel("Average Intensity")
	cont = 0
	x = np.linspace(-1,.2,100)
	y = (-w[0,1]/w[0,2])*x + -w[0,0]/w[0,2]
	for i in data: 
		if(label[cont] == 1.0):
			plt.scatter(i[0],i[1], marker = "*", color = "red")
		else:
			plt.scatter(i[0],i[1], marker = "+",  color = "blue")
		cont = cont + 1
	plt.plot(x,y)
	plt.show()



#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron line directly under should be removed
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc,test_acc


