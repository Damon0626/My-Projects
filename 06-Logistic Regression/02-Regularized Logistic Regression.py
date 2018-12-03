# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-3 上午8:57
# @Email : wwymsn@163.com
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op

class RegularizedLR(object):
	def __init__(self):
		self.init_theta = None

	def loadData(self, path):
		data = pd.read_csv(path, header=None)
		self.data = np.array(data)

	def reshapeData(self):
		m = self.data.shape[0]
		self.x = self.data[:, 0:2]  # (118, 2)
		self.y = self.data[:, 2].reshape((m, 1))  # (118, )-->(118, 1)
		self.init_theta = np.zeros((self.x.shape[1], ))
		self.map_x = self.mapFeature(self.x[:, 0].reshape((m, 1)), self.x[:, 1].reshape((m, 1)))  # (118,28)

	def plotData(self):
		y1_index = np.where(self.y == 1.0)
		x1 = self.x[y1_index[0]]
		y0_index = np.where(self.y == 0.0)
		x0 = self.x[y0_index[0]]
		plt.scatter(x1[:, 0], x1[:, 1], marker='+', color='k')
		plt.scatter(x0[:, 0], x0[:, 1], color='y')
		plt.legend(['y = 1', 'y = 0'])
		plt.xlabel('Microchip Test 1')
		plt.ylabel('Microchip Test 2')
		# plt.show()

	def mapFeature(self, x1, x2):
		# 生成多项式
		out = np.ones((x1.shape[0], 1))
		for i in range(6):
			for j in range(i+2):
				out = np.hstack([out, (x1**(i+1-j)*(x2**j))])
		return out

	def costFunctionReg(self, theta, lanmda):
		m = self.y.shape[0]
		J = (-np.dot(self.y.T, np.log(self.sigmoid(self.map_x.dot(theta))))-np.dot((1-self.y).T, np.log(1-self.sigmoid(self.map_x.dot(theta))))) / m+ (lanmda*np.sum(theta[1::]**2, axis=0))/(2*m)  # 正则化是从j = 1开始的
		return J

	def gradient(self, theta, lanmd):
		m = self.y.shape[0]
		theta = theta.reshape((self.map_x.shape[1], 1))
		grad = np.zeros((self.map_x.shape[1], 1))
		grad[0] = np.dot(self.map_x[:, 0:1].T, (self.sigmoid(self.map_x.dot(theta))-self.y)) / m
		grad[1::] = np.dot(self.map_x[:, 1::].T, (self.sigmoid(self.map_x.dot(theta))-self.y)) / m + lanmd*theta[1::] / m
		# grad = np.dot(self.map_x.T, (self.sigmoid(self.map_x.dot(theta)) - self.y)) / m
		return grad

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def fminunc(self):  # costFunction需要几个参数就传几个,本例中只有一个theta,固x0,也可以利用args=()
		optiTheta = op.minimize(fun=self.costFunctionReg, x0=np.zeros((28, )), args=(100, ), method='TNC', jac=self.gradient)
		return optiTheta  # dict

	def plotDecisionBoundary(self):
		u = np.linspace(-1, 1.5, 50)
		v = np.linspace(-1, 1.5, 50)
		z = np.zeros((len(u), len(v)))
		for i in range(len(u)):
			for j in range(len(v)):
				z[i, j] = self.mapFeatureOne(u[i], v[j]).dot(self.fminunc()['x'])
		z = z.transpose()
		self.plotData()
		plt.title("Lambda = 100")
		plt.contour(u, v, z, 0, colors='red')
		plt.show()

	def mapFeatureOne(self, x1, x2):
		out = [1]
		for i in range(6):
			for j in range(i+2):
				out.append(x1**(i+1-j)*(x2**j))
		return np.array(out).reshape((1, 28))


if __name__ == "__main__":
	path = 'ex2data2.txt'
	RLR = RegularizedLR()
	RLR.loadData(path)
	RLR.reshapeData()
	# RLR.plotData()
	# print(RLR.costFunctionReg(np.zeros((28, 1))))
	# print(RLR.gradient(np.zeros((28, )))[:5])
	# print(RLR.fminunc())
	RLR.plotDecisionBoundary()