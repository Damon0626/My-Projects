# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-2 下午4:49
# @Email : wwymsn@163.com
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import pandas as pd


class LogisticRegression(object):
	def __init__(self):
		self.data = None
		self.init_theta = np.zeros(3, )
		self.test_theta = np.array([-24, 0.2, 0.2])

	def loadData(self, path):
		data = pd.read_csv(path, header=None)
		self.data = np.array(data)

	def reshapeData(self):
		m = self.data.shape[0]
		self.x = self.data[:, 0:2]
		self.y = self.data[:, 2].reshape((m, 1))
		self.x_plus1 = np.hstack([np.ones((100, 1)), self.x])

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def testTheta(self):
		cost = self.costFunction(self.init_theta)
		grad = self.gradient(self.init_theta)
		self.printCostAndGrad(self.init_theta, cost, grad)

		cost = self.costFunction(self.test_theta)
		grad = self.gradient(self.test_theta)
		self.printCostAndGrad(self.test_theta, cost, grad)

	def printCostAndGrad(self, theta, cost, grad):
		print("Cost at theta {} \n{}".format(theta, cost))
		print("Gradient at theta {}\n{}".format(theta, grad))

	def costFunction(self, theta):
		m = len(self.y)
		J = np.sum(-np.dot(self.y.T, np.log(self.sigmoid(self.x_plus1.dot(theta))))\
			-np.dot((1-self.y).T, np.log(1-self.sigmoid(self.x_plus1.dot(theta)))), axis=0) / m
		return J

	def gradient(self, theta):
		m = len(self.y)
		theta = theta.reshape((3, 1))
		grad = np.dot(self.x_plus1.T, (self.sigmoid(self.x_plus1.dot(theta))-self.y)) / m
		return grad

	def plottingData(self):
		y1_index = np.where(self.y == 1.0)
		x1 = self.x[y1_index[0]]
		y0_index = np.where(self.y == 0.0)
		x0 = self.x[y0_index[0]]
		plt.scatter(x1[:, 0], x1[:, 1], marker='+', color='k')
		plt.scatter(x0[:, 0], x0[:, 1], color='y')
		plt.xlabel('Exam 1 score')
		plt.ylabel('Exam 2 score')

	def fminunc(self):  # costFunction需要几个参数就传几个,本例中只有一个theta,固x0,也可以利用args=()
		optiTheta = op.minimize(fun=self.costFunction, x0=self.init_theta, method='TNC', jac=self.gradient)
		return optiTheta  # dict

	def plotRegLine(self):  # 两点确定一条直线
		self.opti_theta = self.fminunc()['x']
		plot_x = [np.min(self.x[:, 0]), np.max(self.x[:, 1])]  # [A, B]
		plot_y = [-(self.opti_theta[0] + self.opti_theta[1]*x)/self.opti_theta[2] for x in plot_x]  # [A, B]
		self.plottingData()
		plt.plot(plot_x, plot_y)
		plt.legend(['Decision Boundary', 'Admitted', 'Not admitted'])

		plt.show()

	def predictAndAccuracies(self):
		prob = self.sigmoid(np.array(self.opti_theta).reshape(1, 3).dot(np.array([[1], [45], [85]])))
		print('For a student with scores with 45 and 85, we predict an admission probability of %f' % prob)
		accuracy = np.mean(self.predict0_1(self.opti_theta) == self.y)*100
		print('Train Accuracy: %f' % accuracy)

	def predict0_1(self, theta):
		self.p = np.zeros((100, 1))
		self.p = self.sigmoid(self.x_plus1.dot(np.array(theta).reshape(3, 1)))
		for i in range(len(self.p)):
			if self.p[i] < 0.5:
				self.p[i] = 0
			else:
				self.p[i] = 1
			i += 1
		return self.p


if __name__ == '__main__':
	path = 'ex2data1.txt'
	LR = LogisticRegression()
	LR.loadData(path)
	LR.reshapeData()
	# LR.plottingData()
	LR.testTheta()
	print('*'*10, 'Testing Over!', '*'*10)
	print(LR.fminunc())
	print('Cost at theta found by fminunc is: \n{}'.format(LR.fminunc()['fun']))
	print('Theta found by fminunc is: \n{}'.format(LR.fminunc()['x']))
	LR.plotRegLine()
	LR.predictAndAccuracies()
	'''
	Cost at theta [ 0.  0.  0.] 
	0.6931471805599453
	Gradient at theta [ 0.  0.  0.]
	[[ -0.1       ]
	 [-12.00921659]
	 [-11.26284221]]
	 
	Cost at theta [-24.    0.2   0.2] 
	0.21833019382659785
	Gradient at theta [-24.    0.2   0.2]
	[[ 0.04290299]
	 [ 2.56623412]
	 [ 2.64679737]]

	********** Testing Over! **********
		fun: 0.20349770158947458
		jac: array([[  9.14708729e-09],
			[  9.94227578e-08],
			[  4.83045640e-07]])
		message: 'Local minimum reached (|pg| ~= 0)'
		nfev: 36
		 nit: 17
		status: 0
		success: True
		x: array([-25.16131863,   0.20623159,   0.20147149])

	Cost at theta found by fminunc is: 
	0.20349770158947458

	Theta found by fminunc is: 
	[-25.16131863   0.20623159   0.20147149]

	For a student with scores with 45 and 85, we predict an admission probability of 0.776291
	Train Accuracy: 89.000000
	'''