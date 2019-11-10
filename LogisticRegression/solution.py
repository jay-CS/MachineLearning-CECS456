import numpy as np 
import math
from helper import *
'''
Homework2: logistic regression classifier
'''

def sigmoid(s):
	return (1/(1 + np.exp(-s)))


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
	#My solution
	# for i in range(len(label)):
	# 	if(label[i] == -1.0):
	# 		target[i] = 0
	# w = np.zeros((3,1))
	# for i in range (max_iter):
	# 	y_predict = sigmoid(np.dot(data[:],w))
		
	# 	#print(y_predict.shape)
	# 	dw = np.dot(data[:].transpose(),(y_predict - target))
	# 	w -= learning_rate * dw

	n = len(label)
	w = np.zeros((3,1))
	gd = 0
	for i in range(max_iter):
		for j in range(n):
			theta = -1*(label[j]* np.dot(w.transpose(),data[j]))
			s = sigmoid(theta)
			gd += (np.dot(label[j],data[j]))*s
		gd *= -(1/n) * learning_rate
		w[0] = w[0][0] - gd[0]
		w[1] = w[1][0] - gd[1]
		w[2] = w[2][0] - gd[2]
	print("Ouput:",w)
	return w

	
	# gd = 0
	# for i in range(max_iter):
	# 	for x in range(len(label)):
	# 		# s =  label[x]*(np.dot(data[x],w))
	# 		# theta = sigmoid(-s)
	# 		ss = w[0]*data[x][0] + w[1]*data[x][1] + w[2]*data[x][2]
	# 		gd +=  ((label[x] * data[x][0])/ (1+np.exp(label[x] * ss)))
	# 	gd *= (-1/len(label))
	# 	w -= learning_rate*gd





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
	pass


def accuracy(x, y, w):
    # '''
    # This function is used to compute accuracy of a logsitic regression model.
    
    # Args:
    # x: input data with shape (n, d), where n represents total data samples and d represents
    #     total feature numbers of a certain data sample.
    # y: corresponding label of x with shape(n, 1), where n represents total data samples.
    # w: the seperator learnt from logistic regression function with shape (d, 1),
    #     where d represents total feature numbers of a certain data sample.

    # Return 
    #     accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
    #     which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
	
	# n = 0
	# y_pred = sigmoid(np.dot(x[:],w))
	# y_pred1 = [1.0 if yi > .5 else 0.0 for yi in y_pred]
	# for i in range(len(y)):
	# 	if(y[i] == -1.0):
	# 		y[i] = 0.0
	# for entry in y_pred1:
	# 	if(entry != y[n]):
	# 		mistakes += 1
	# 	n +=1 


	# n = 0
	# mistakes = 0 
	# y_true = np.ones((len(y),1))
	# for i in range(len(y)):
	# 	if(y[i] == -1.0):
	# 		y_true[i] = 0
	# for entry in x:
	# 	y_pred = 1.0 if np.dot(entry,w) > .5 else 0.0
	# 	if(y_pred != y_true[n]):
	# 		mistakes += 1
	# 	n += 1
	# print("N: ",n)
	# print((n - mistakes)/(len(y)))
	# return (n-mistakes)/(len(y))
	n = len(y)
	count = 0
	mistakes = 0
	y_pred = 0
	for entry in x:
		y_pred = 1.0 if sigmoid(np.dot(entry,w)) > .5 else -1.0
		if(y_pred != y[count]):
			mistakes += 1
		count += 1
	print("Accuracy: %.3f%" % ((n-mistakes)/n))
	# for i in range(n):
	# 	print("Predicted: ",y_pred[i],"Actual:",y[i])
	return (n-mistakes)/n

	# count = 0
	# mistakes = 0
	# n = len(y)
	# y_new = np.ones((n,1))
	# for i in range(n):
	# 	if(y[i] == -1.0):
	# 		y_new[i] = 0
	# for entry in x:
	# 	y_pred = 1.0 if np.dot(entry,w) > .5 else 0.0
	# 	if(y_pred != y_new[count]):
	# 		mistakes += 1
	# 	count += 1
	

