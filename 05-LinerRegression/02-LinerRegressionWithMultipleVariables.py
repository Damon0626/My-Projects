# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-2 上午9:18
# @Email : wwymsn@163.com
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt


class LinerRegwithMultiVariables(object):
	def __init__(self):
		self.rows = 0
		self.data = np.zeros((47, 3))
		self.alpha = 0.01
		self.iters = 400
		self.theta = np.zeros((3, 1))

	def loadData(self, path):
		try:
			with open(path) as f:
				for line in f.readlines():
					self.data[self.rows] = line.strip('\n').split(',')
					self.rows += 1
			f.close()
			self.y = self.data[:, 2].reshape(47, 1)  # 提取一列，(47, ) --> (47, 1)
			self.x = self.data[:, 0:2].reshape(47, 2)  # (47, 2)
		except:
			print("Error: Not have the %s" % path)

	def first10Examples(self):
		print('First 10 examples from the dataset:')
		for i in range(10):
			print('x = {}, y = {}'.format(self.x[i, :], self.y[i]))

	def featureNormalize(self):
		mu = np.mean(self.x, axis=0)
		sigma = np.std(self.x, axis=0, ddof=1)  # 注意ddof参数，必须是1
		self.x_norm = (self.x - mu)/sigma
		self.x = np.hstack([np.ones((47, 1)), self.x_norm])

	def gradientDescentMulti(self):
		m = len(self.y)
		self.J_history = np.zeros((self.iters, 1))

		for i in range(self.iters):
			self.theta = self.theta-self.alpha/m*np.dot((np.dot(self.x, self.theta)-self.y).transpose(), self.x).transpose()
			self.J_history[i] = self.computeCostMulti()
		return self.theta

	def computeCostMulti(self):
		m = len(self.y)
		J = 1/(2*m)*np.sum((np.dot(self.x, self.theta) - self.y)**2)
		return J

	def convergenceGraph(self):
		plt.plot([x for x in range(400)], self.J_history, 'b')
		plt.xlabel('Number of iterations')
		plt.ylabel('Cost J')
		plt.show()

	def estimatePrice(self, squre, house):
		price = np.dot(self.theta.T, [[1], [squre], [house]])
		price = price.tolist()
		print('Predicted price of a {} sq-ft, {} br house(using gradient descent) is ${[0][0]}'.format(squre, house, price))

	def normalEquations(self):
		self.x = np.hstack([np.ones((47, 1)), self.x])
		self.theta = np.dot(np.dot(np.mat(np.dot(self.x.T, self.x)).I, self.x.T), self.y)
		print(self.theta)

if __name__ == '__main__':
	path = 'ex1data2.txt'
	LRMV = LinerRegwithMultiVariables()
	LRMV.loadData(path)
	LRMV.first10Examples()
	LRMV.featureNormalize()
	print(LRMV.gradientDescentMulti())  # 梯度下降方法计算theta
	LRMV.convergenceGraph()
	LRMV.estimatePrice(1650, 3)  # 计算房价

	# 利用Normal Equations计算theta, 不需要特征归一化, 而且没有直到收敛的循环操作
	NE = LinerRegwithMultiVariables()
	NE.loadData(path)
	NE.normalEquations()
	NE.estimatePrice(1650, 3)
