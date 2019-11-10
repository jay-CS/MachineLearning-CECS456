import numpy as np 
from helper import *
import math 
import random
'''
Homework2: logistic regression classifier
'''

def sigmoid(s):
	return (1/(1+np.exp(-s)))

def logistic_regression(data, label, max_iter, learning_rate):
	'''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
	'''
	n = len(label)
	w = np.zeros((len(data[0]),1))
	w = np.transpose(w)
	for i in range(max_iter):
		gd = np.zeros((1,len(data[0])))
		for j in range(n):
			gd += (label[j]*data[j]) / (1+np.exp(label[j] * np.dot(w,data[j])))	
		gd *= (-1/n)
		w -= learning_rate*gd
	#print(w)
	return np.transpose(w)


def logistic_regressionSGD(data,label,max_iter,learning_rate):
	n = len(label)
	w = np.zeros((3,1))
	w = np.transpose(w)
	gd = np.zeros((1,3))
	for i in range(max_iter):
		select = random.randint(0,max_iter)
		gd = (-1*label[select]*data[select]) / (1+np.exp(label[select] * np.dot(w,data[select])))	
		w -= learning_rate*gd
	#print(w)
	return np.transpose(w)

	


def thirdorder(data):
	'''
	This function is used for a 3rd order polynomial transform of the data.
	Args:
	data: input data with shape (:, 3) the first dimension represents 
		  total samples (training: 1561; testing: 424) and the 
		  second dimesion represents total features.

	Return:
		result: A numpy array format new data with shape (:,10), which using 
		a 3rd order polynomial transformation to extend the feature numbers 
		from 3 to 10. 
		The first dimension represents total samples (training: 1561; testing: 424) 
		and the second dimesion represents total features.
	'''
	z_x = np.zeros((len(data),10))
	for i in range(len(data)):
		z_x[i] = [1,data[i][0],data[i][1], (data[i][0]**2) , data[i][0] * data[i][1] ,(data[i][1]**2) ,(data[i][0]**3), (data[i][0]**2) * data[i][1] 
		,(data[i][0]) * (data[i][1]**2), data[i][1]**3 ]
	return z_x


def accuracy(x, y, w):
    # '''
    # This function is used to compute accuracy of a logsitic regression model.
    
    # Args:
    # x: input data with shape (n, d), where n represents total data samples and d represents
    #     total feature numbers of a certain data sample.
    # y: corresponding label of x with shape(n, 1), where n represents total data samples.
    # w: the seperator learned from logistic regression function with shape (d, 1),
    #     where d represents the total feature numbers of a certain data sample.

    # Return 
    #     accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
    #     which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    # '''mistakes = 0
	count = 0
	mistakes = 0
	n = len(y)
	w = np.transpose(w)
	for z in range(n):
		y_pred = 1.0 if sigmoid(np.dot(w,x[z])) > .5 else -1.0
		if(y_pred != y[count]):
			mistakes += 1
		count += 1
	return (n-mistakes)/n