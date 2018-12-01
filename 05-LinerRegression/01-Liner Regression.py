# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-1 上午10:00
# @Email : wwymsn@163.com
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LinerRegression(object):
	def __init__(self):
		self.data = np.zeros((97, 2))
		self.rows = 0
		self.theta = np.zeros((2, 1))
		self.iters = 1500
		self.alpha = 0.01

	def readData(self, path):
		try:
			with open(path) as f:
				for line in f.readlines():
					self.data[self.rows] = line.strip('\n').split(',')
					self.rows += 1
			f.close()
			self.y = self.data[:, 1].reshape(97, 1)  # 提取一列，(97, ) --> (97, 1)
			self.x = np.hstack([np.ones((97, 1)), self.data[:, 0].reshape(97, 1)])  # 添加一列１ (97, 2)
		except:
			print("Error: Not have the %s" % path)

	def computeCost(self, theta):
		m = len(self.y)
		J = 1/(2*m)*np.sum((np.dot(self.x, theta)-self.y)**2)  # costFunction
		return J

	def gradientDescent(self):
		m = len(self.y)
		J_history = np.zeros((self.iters, 1))

		for i in range(self.iters):
			self.theta -= self.alpha/m*(np.dot((np.dot(self.x, self.theta) - self.y).transpose(), self.x)).transpose()
			J_history[i] = self.computeCost(self.theta)

		return self.theta

	def regressionLine(self):
		plt.scatter(self.data[:, 0], self.data[:, 1], marker='+', color='r')  # 画出初始点图
		plt.title('LR')
		plt.plot(self.x[:, 1], np.dot(self.x, self.gradientDescent()))
		plt.legend(['Liner Regression', 'Training Data'], loc='lower right')
		plt.xlabel('Population of Cith in 10,000s')
		plt.ylabel('Profit in $10,000s')
		plt.show()

	def costJSurface(self):
		self.theta0_vals = np.linspace(-10, 10, 100).reshape(100, 1)
		self.theta1_vals = np.linspace(-1, 4, 100).reshape(100, 1)

		J_vals = np.zeros((len(self.theta0_vals), len(self.theta1_vals)))  # 100*100

		for i in range(J_vals.shape[0]):
			for j in range(J_vals.shape[1]):
				t = np.array([self.theta0_vals[i], self.theta1_vals[j]])
				J_vals[i, j] = self.computeCost(t)
		self.J_vals = J_vals.transpose()
		fig = plt.figure()
		ax = Axes3D(fig)
		self.theta0_vals, self.theta1_vals = np.meshgrid(self.theta0_vals, self.theta1_vals)
		p1 = ax.plot_surface(self.theta0_vals, self.theta1_vals, self.J_vals, cmap='rainbow')
		plt.title("Surface")
		plt.xlabel("theta0")
		plt.ylabel("theta1")
		plt.colorbar(p1)
		plt.show()

	def costJcontours(self):
		plt.contourf(self.theta0_vals, self.theta1_vals, self.J_vals, 15)  # 按照值分为15个区域
		plt.plot(self.gradientDescent()[0], self.gradientDescent()[1], marker='+')  # 求得的最优解位置
		plt.title("Contour, showing minimum")
		plt.xlabel("theta0")
		plt.ylabel("theta1")

		plt.show()


if __name__ == "__main__":
	LR = LinerRegression()
	LR.readData('ex1data1.txt')

	print('-'*10+'测试theta[0, 0]:预期结果32.07'+'-'*10)
	theta = np.array([[0], [0]])
	print("求得损失函数J结果：%f" % LR.computeCost(theta))

	print('-' * 10 + '测试theta[-1, 2]:预期结果54.24' + '-' * 10)
	theta = np.array([[-1], [2]])
	print("求得损失函数J结果：%f" % LR.computeCost(theta))

	LR.regressionLine()  # 绘制点和拟合直线
	LR.costJSurface()
	LR.costJcontours()

