# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-8 下午8:41
# @Email : wwymsn@163.com
# @Software: PyCharm

import scipy.io as scio
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

class NNLearning(object):
	def __init__(self):
		self.input_layer_size = 400
		self.hidden_layer_size = 25
		self.num_labels = 10

	def loadData(self, path):
		self.data = scio.loadmat(path)
		self.x = self.data["X"]  # (5000, 400)  # 原100训练
		self.y = self.data["y"]  # (5000, 1)
		index = random.sample([i for i in range(5000)], 100)  # 随机100个没有重复的数字
		self.pics = self.x[index, :]  # (100, 400)

	def loadWeights(self, path):
		weights = scio.loadmat(path)
		self.theta1 = weights['Theta1']  # 25*401
		self.theta2 = weights['Theta2']  # 10*26

	def display100Data(self):
		example_width = int(np.sqrt(self.pics.shape[1]))  # 每张图片的宽
		example_hight = self.pics.shape[1] // example_width

		display_rows = int(np.sqrt(self.pics.shape[0]))  # 每行显示几张图片
		display_cols = self.pics.shape[0] // display_rows
		# print(self.pics[45, :])
		display_array = np.ones((1+display_rows*(example_hight+1), 1+display_cols*(example_width+1)))*200
		curr_ex = 0  # 当前每行张数
		for i in range(display_rows):
			for j in range(display_cols):
				if curr_ex >= self.pics.shape[0]:
					break
				max_val = np.max(np.abs(self.pics[curr_ex, :]))
				display_array[1+j*(example_hight+1):(j+1)*(example_hight+1), 1+i*(example_width+1):(i+1)*(example_width+1)] = \
					self.pics[curr_ex, :].reshape((20, 20)).transpose()/max_val*255
				curr_ex += 1

			if curr_ex >= self.pics.shape[0]:
				break
		plt.xticks([])
		plt.yticks([])
		plt.imshow(display_array, cmap='gray')
		plt.show()

	def nnCostFunction(self, theta, x, y, lamda):
		m = x.shape[0]
		theta1 = np.reshape(theta[:self.hidden_layer_size*(self.input_layer_size+1)], (self.hidden_layer_size, self.input_layer_size+1))
		theta2 = np.reshape(theta[self.hidden_layer_size*(self.input_layer_size+1)::], (self.num_labels, self.hidden_layer_size+1))
		y = self.handleYtoOne(y)
		a1 = np.hstack([np.ones((m, 1)), x])  # 5000, 401
		z2 = a1.dot(theta1.T)  # 5000*25
		a2 = self.sigmoid(z2)
		n = a2.shape[0]  # 5000
		a2 = np.hstack([np.ones((n, 1)), a2])  # 5000*26
		z3 = a2.dot(theta2.T)
		a3 = self.sigmoid(z3)  # 5000*10

		J = np.sum(np.sum(-y*np.log(a3)-(1-y)*np.log(1-a3), axis=0))/m

		regularized1 = np.sum(np.sum(theta1[:, 1::]**2, axis=0))
		regularized2 = np.sum(np.sum(theta2[:, 1::]**2, axis=0))
		regularized = lamda/(2*m)*(regularized1 + regularized2)
		return J + regularized

	def nnGradient(self, theta, x, y, lamda):
		m = x.shape[0]
		theta1 = np.reshape(theta[:self.hidden_layer_size*(self.input_layer_size+1)], (self.hidden_layer_size, self.input_layer_size+1))
		theta2 = np.reshape(theta[self.hidden_layer_size*(self.input_layer_size+1)::], (self.num_labels, self.hidden_layer_size+1))
		y = self.handleYtoOne(y)
		a1 = np.hstack([np.ones((m, 1)), x])  # 5000, 401
		z2 = a1.dot(theta1.T)  # 5000*25
		a2 = self.sigmoid(z2)
		n = a2.shape[0]  # 5000
		a2 = np.hstack([np.ones((n, 1)), a2])  # 5000*26
		z3 = a2.dot(theta2.T)
		a3 = self.sigmoid(z3)  # 5000*10

		delta3 = a3 - y
		delta2 = delta3.dot(theta2)
		delta2 = delta2[:, 1::]
		delta2 = delta2*self.sigmoidGradient(z2)  # 5000*25
		Delta1 = np.zeros(theta1.shape)
		Delta2 = np.zeros(theta2.shape)

		Delta1 = Delta1 + delta2.T.dot(a1)
		Delta2 = Delta2 + delta3.T.dot(a2)

		Theta1_grad = 1/m*Delta1
		Theta2_grad = 1/m*Delta2

		Regularized_T1 = lamda/m*theta1
		Regularized_T2 = lamda/m*theta2
		Regularized_T1[:, 0] = np.zeros((Regularized_T1.shape[0], ))
		Regularized_T2[:, 0] = np.zeros((Regularized_T2.shape[0], ))

		Theta1_grad += Regularized_T1
		Theta2_grad += Regularized_T2
		grade = np.hstack([Theta1_grad.flatten(), Theta2_grad.flatten()])
		return grade

	def handleYtoOne(self, y):
		handle_y = np.zeros((len(y), self.num_labels))
		for i in range(self.num_labels):
			if i+1 == self.num_labels:
				handle_y[len(y)//self.num_labels*0:len(y)//self.num_labels*1, i] = 1
			else:
				handle_y[len(y)//self.num_labels*(i+1):len(y)//self.num_labels*(i+2), i] = 1

		return handle_y

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def sigmoidGradient(self, z):
		g = self.sigmoid(z)*(1-self.sigmoid(z))
		return g

	def randInitializeWeights(self, L_in, L_out):
		epsion_init = 0.12
		W = np.random.rand(L_out, 1+L_in)*2*epsion_init-epsion_init
		return W

	def checkNNGradients(self, lamda):
		self.input_layer_size = 3
		self.hidden_layer_size = 5
		self.num_labels = 3
		m = 5
		theta1 = self.debugInitializeWeights(self.hidden_layer_size, self.input_layer_size)
		theta2 = self.debugInitializeWeights(self.num_labels, self.hidden_layer_size)
		x = self.debugInitializeWeights(m, self.input_layer_size-1)
		y = 1 + np.mod([i+1 for i in range(m)], self.num_labels).T
		theta = np.hstack([theta1.flatten(), theta2.flatten()])
		cost = self.nnCostFunction(theta, x, y, lamda)
		grad = self.nnGradient(theta, x, y, lamda)
		numgrad = self.computeNumericalGradient(theta, x, y, lamda)
		# 求解最大奇异值
		diff = max((numgrad-grad)/(numgrad+grad))
		print(np.hstack([grad.reshape(-1, 1), numgrad.reshape(-1, 1)]))
		print("Relative Difference:", diff)

	def debugInitializeWeights(self, fan_out, fan_in):
		w = np.reshape(np.sin([i+1 for i in range(fan_out*(1+fan_in))]), (fan_out, 1+fan_in))/10
		return w

	def computeNumericalGradient(self, theta, x, y, lamda):  # (f(x+delta)-f(x-delta))/(2*delta)
		e = 0.0001
		numgrad = np.zeros(theta.shape)
		perturb = np.zeros(theta.shape)
		for i in range(theta.size):
			perturb[i] = e
			loss1 = self.nnCostFunction(theta - perturb, x, y, lamda)
			loss2 = self.nnCostFunction(theta + perturb, x, y, lamda)
			numgrad[i] = ((np.array(loss2) - np.array(loss1))/(2*e))
			perturb[i] = 0
		return numgrad


if __name__ == '__main__':
	datapath = 'ex4data1.mat'
	weightspath = 'ex4weights.mat'
	NNL = NNLearning()
	NNL.loadData(datapath)
	NNL.loadWeights(weightspath)
	# NNL.display100Data()
	NNL.checkNNGradients(3)