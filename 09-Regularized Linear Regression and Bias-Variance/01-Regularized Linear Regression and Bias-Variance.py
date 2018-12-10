# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-10  下午9:04
# @Email : wwymsn@163.com
# @Software: PyCharm


import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op


class BiasAndVariance(object):
	def __init__(self):
		pass

	def loadData(self, path):
		self.data = scio.loadmat(path)

	def plotXY(self):
		X = self.data['X']
		Y = self.data['y']
		plt.scatter(X, Y, marker='x', color='r')
		plt.xlabel('Change in water level(x)')
		plt.ylabel('Water flowing out of the dam (y)')
		plt.show()

	def linearRegCostFunction(self, theta, x, y, lamda):
		x = np.hstack([np.ones((x.shape[0], 1)), x])
		m = x.shape[0]
		theta = theta.reshape((x.shape[1], 1))
		cost = np.sum((x.dot(theta)-y)**2)/(2*m)
		regular = lamda/(2*m)*np.sum(theta[1::]**2)
		J = cost + regular
		return J

	def linearRegGradient(self, theta, x, y, lamda):
		x = np.hstack([np.ones((x.shape[0], 1)), x])
		m = y.shape[0]
		theta = theta.reshape((x.shape[1], 1))
		grad = np.zeros((x.shape[1], 1))
		grad[0] = 1/m*(x[:, 0:1].T.dot(x.dot(theta)-y))
		grad[1::] = 1/m*(x[:, 1::].T.dot(x.dot(theta)-y)) + lamda/m*theta[1::]
		return grad

	def caculateJandGra(self):
		theta1 = np.array([[1], [1]])
		print("="*10+"Regularized Linear Reression Cost"+"="*10)
		x = self.data['X']
		y = self.data['y']
		lamda = 1
		J = self.linearRegCostFunction(theta1, x, y, lamda)
		print("Cost at theta=[1;1] is {}".format(J)+"\n(this value should be about 303.993192)\n")
		print("="*10+"Regularized Linear Reression Gradient"+"="*10)
		grad = self.linearRegGradient(theta1, x, y, lamda)
		print("Gradient at theta=[1;1] is {0}{1}".format(grad[0], grad[1])+
		      "\n(this value should be about [-15.303016;598.250744])")

	def trainLinearReg(self, x, y, lamda):
		initial_theta = np.zeros((x.shape[1]+1, ))
		fmin = op.minimize(fun=self.linearRegCostFunction, x0=initial_theta, args=(x, y, lamda), method='TNC', jac=self.linearRegGradient)
		theta = fmin['x']
		# x_plot = np.hstack([np.ones((x.shape[0], 1)), x])
		# plt.scatter(x, y, marker='+', color='r')
		# plt.plot(x, x_plot.dot(theta), '--', linewidth=2)
		# plt.xlabel('Change in water level (x)')
		# plt.ylabel('Water flowing out of the dam (y)')
		# plt.legend(['Trained', 'Original'])
		# plt.show()
		return theta

	def learningCurve(self):
		x = self.data['X']
		y = self.data['y']
		xval = self.data['Xval']  # (21, 1)
		yval = self.data['yval']
		m = x.shape[0]
		error_train = np.zeros((m, 1))
		error_val = np.zeros((m, 1))
		print("Training Examples\tTrain Error\tCross Validation Error\n")
		for i in range(m):
			theta = self.trainLinearReg(x[:1+i, :], y[:1+i], 0)
			error_train[i] = self.linearRegCostFunction(theta, x[:1+i, :], y[:1+i], 0)
			error_val[i] = self.linearRegCostFunction(theta, xval, yval, 0)
			print("\t\t%d\t\t\t%f\t\t%f\n"%(i, error_train[i], error_val[i]))
		plt.xlim([0, 13])
		plt.ylim([0, 150])
		plt.plot([i for i in range(12)], error_train, 'r')
		plt.plot([i for i in range(12)], error_val, 'b')
		plt.title('Learning curve for linear regression')
		plt.xlabel('Number of training examples')
		plt.ylabel('Error')
		plt.legend(['Train', 'Cross Validation'])
		plt.show()


if __name__ == "__main__":
	path = 'ex5data1.mat'
	BAV = BiasAndVariance()
	BAV.loadData(path)
	# BAV.plotXY()
	# BAV.caculateJandGra()
	# BAV.trainLinearReg()
	# BAV.learningCurve()
	BAV.learningCurve()