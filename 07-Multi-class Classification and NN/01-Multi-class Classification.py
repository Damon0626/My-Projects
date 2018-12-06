# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-4 下午10:46
# @Email : wwymsn@163.com
# @Software: PyCharm

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import random


class MultiClassClassification(object):
	def __init__(self):
		self.num_lables = 10
		self.init_theta = np.zeros((401, ))

	def loadData(self, path):
		self.data = scio.loadmat(path)
		# self.x = self.data["X"]  # (5000, 400)  # 原100训练
		# self.y = self.data["y"]  # (5000, 1)
		index = random.sample([i for i in range(5000)], 4000)  # 80%training 20%testing
		self.train_x = self.data["X"][index, :]
		self.train_y = self.data["y"][index, :]
		self.test_x = np.delete(self.data["X"], index, 0)
		self.test_y = np.delete(self.data["y"], index, 0)
		print(self.test_y)
	def display100Pics(self):
		# 5000张随机找到100张
		index = np.random.randint(1, 4000, size=100)
		self.pics = self.train_x[index, :]  # (100, 400)

	def displayData(self):
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

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def logisticRegressionTest(self):
		theta_t = np.array([[-2], [-1], [1], [2]])
		x_t = np.hstack([np.ones((5, 1)), np.arange(1, 16).reshape(3, 5).T/10])
		y_t = np.array([[1], [0], [1], [0], [1]])
		lambda_t = 3
		print(self.lrCostFunction(theta_t, x_t, y_t, lambda_t))
		print(self.lrGradient(theta_t, x_t, y_t, lambda_t))

	def lrCostFunction(self, theta, x, y, lamda):
		m = y.shape[0]
		J = (-np.dot(y.T, np.log(self.sigmoid(x.dot(theta))))-np.dot((1-y).T, np.log(1-self.sigmoid(x.dot(theta))))) / m+ (lamda*np.sum(theta[1::]**2, axis=0))/(2*m)  # 正则化是从j = 1开始的
		return J

	def lrGradient(self, theta, x, y, lamda):
		m = y.shape[0]
		theta = theta.reshape((x.shape[1], 1))
		grad = np.zeros((x.shape[1], 1))
		grad[0] = np.dot(x[:, 0:1].T, (self.sigmoid(x.dot(theta))-y)) / m
		grad[1::] = np.dot(x[:, 1::].T, (self.sigmoid(x.dot(theta))-y)) / m + lamda*theta[1::] / m
		return grad

	def fmini(self):
		x = np.hstack([np.ones((self.train_x.shape[0], 1)), self.train_x])  # (5000, 401)
		y = self.train_y
		# fmincg = op.fmin_cg(f=self.lrCostFunction, x0=self.init_theta, fprime=self.lrGradient, args=(x, np.array(y==(1+1), np.int), 0.1)) #报错
		self.optiTheta = np.zeros((10, 401))
		for i in range(10):
			fmini = op.minimize(fun=self.lrCostFunction, x0=self.init_theta, args=(x, np.array(y==(i+1), np.int), 0.1), method='TNC', jac=self.lrGradient)
			print("训练第%d部分"%(i+1))
			self.optiTheta[i, :] = fmini['x']
		return self.optiTheta

	def predictOneVsAll(self):
		x = np.hstack([np.ones((self.test_x.shape[0], 1)), self.test_x])  # (1000, 401)
		position = np.argmax(self.sigmoid(x.dot(self.optiTheta.T)), axis=1) + 1
		accuracy = np.mean(position.reshape(1000, 1) == self.test_y)*100
		# print(position[:50])
		# print("100%traingset accuracy:{}".format(accuracy))  # 96.46%
		print("In 80%traing set, 20%testing set condition, accuracy is  {}".format(accuracy))  # 89.1%


if __name__ == "__main__":
	path = 'ex3data1.mat'
	MCC = MultiClassClassification()
	MCC.loadData(path)
	MCC.display100Pics()
	MCC.displayData()
	MCC.logisticRegressionTest()  # 对给定的值进行测试
	MCC.fmini()
	MCC.predictOneVsAll()