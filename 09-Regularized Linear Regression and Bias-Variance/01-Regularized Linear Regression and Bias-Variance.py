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
		self.x = self.data['X']
		self.y = self.data['y']
		self.xval = self.data['Xval']
		self.yval = self.data['yval']
		self.xtest = self.data['Xtest']
		self.ytest = self.data['ytest']
		self.x_plus_one = np.hstack([np.ones((self.x.shape[0], 1)), self.x])
		self.xval_plus_one = np.hstack([np.ones((self.xval.shape[0], 1)), self.xval])


	def plotXY(self, x, y):
		plt.scatter(x, y, marker='x', color='r')
		plt.xlabel('Change in water level(x)')
		plt.ylabel('Water flowing out of the dam (y)')
		plt.show()

	def linearRegCostFunction(self, theta, x, y, lamda):
		m = x.shape[0]
		theta = theta.reshape((x.shape[1], 1))
		cost = np.sum((x.dot(theta)-y)**2)/(2*m)
		regular = lamda/(2*m)*np.sum(theta[1::]**2)
		J = cost + regular
		return J

	def linearRegGradient(self, theta, x, y, lamda):
		m = y.shape[0]
		theta = theta.reshape((x.shape[1], 1))
		grad = np.zeros((x.shape[1], 1))
		grad[0] = 1/m*(x[:, 0:1].T.dot(x.dot(theta)-y))
		grad[1::] = 1/m*(x[:, 1::].T.dot(x.dot(theta)-y)) + lamda/m*theta[1::]
		return grad

	def caculateJandGra(self, x, y, lamda):
		theta1 = np.array([[1], [1]])
		print("="*10+"Regularized Linear Reression Cost"+"="*10)
		J = self.linearRegCostFunction(theta1, x, y, lamda)
		print("Cost at theta=[1;1] is {}".format(J)+"\n(this value should be about 303.993192)\n")
		print("="*10+"Regularized Linear Reression Gradient"+"="*10)
		grad = self.linearRegGradient(theta1, x, y, lamda)
		print("Gradient at theta=[1;1] is {0}{1}".format(grad[0], grad[1])+
		      "\n(this value should be about [-15.303016;598.250744])")

	def trainLinearReg(self, x, y, lamda):
		initial_theta = np.zeros((x.shape[1], ))
		fmin = op.minimize(fun=self.linearRegCostFunction, x0=initial_theta, args=(x, y, lamda), method='TNC', jac=self.linearRegGradient)
		theta = fmin['x']
		return theta

	def plotTrainingLine(self, x, y):
		theta = self.trainLinearReg(x, y, 0)
		plt.scatter(self.x, self.y, marker='+', color='r')
		plt.plot(self.x, x.dot(theta), '--', linewidth=2)
		plt.xlabel('Change in water level (x)')
		plt.ylabel('Water flowing out of the dam (y)')
		plt.legend(['Trained', 'Original'])
		plt.show()

	def learningCurve(self, x, y, xval, yval, lamda):
		m = x.shape[0]
		error_train = np.zeros((m, 1))
		error_val = np.zeros((m, 1))
		print("Training Examples\tTrain Error\tCross Validation Error\n")
		for i in range(m):
			theta = self.trainLinearReg(x[:1+i, :], y[:1+i], lamda)
			error_train[i] = self.linearRegCostFunction(theta, x[:1+i, :], y[:1+i], 0)
			error_val[i] = self.linearRegCostFunction(theta, xval, yval, 0)
			print("\t\t%d\t\t\t%f\t\t%f\n"%(i, error_train[i], error_val[i]))
		return [error_train, error_val]

	def plotLinerRCurve(self):
		error_train, error_val = self.learningCurve(self.x_plus_one, self.y, self.xval_plus_one, self.yval, 0)
		plt.xlim([0, 13])
		plt.ylim([0, 150])
		plt.plot([i for i in range(12)], error_train, 'r')
		plt.plot([i for i in range(12)], error_val, 'b')
		plt.title('Learning curve for linear regression')
		plt.xlabel('Number of training examples')
		plt.ylabel('Error')
		plt.legend(['Train', 'Cross Validation'])
		plt.show()

	def plotPolyNomialLearningCurve(self):
		error_train, error_val = self.learningCurve(self.x_poly, self.y, self.x_poly_val, self.yval, 0)  # 0 1都试下
		plt.xlim([0, 13])
		plt.ylim([0, 100])
		plt.plot([i for i in range(12)], error_train, 'r')
		plt.plot([i for i in range(12)], error_val, 'b')
		plt.title('Polynomial Regression Learning Curve(lambda=1.00)')
		plt.xlabel('Number of training examples')
		plt.ylabel('Error')
		plt.legend(['Train', 'Cross Validation'])
		plt.show()

	def plotValidationCurveForLambdas(self):
		lamda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape((-1, 1))
		print(lamda_vec.shape)
		error_train, error_val = self.validationCurveForLamdas(self.x_poly, self.y, self.x_poly_val, self.yval, lamda_vec)
		plt.plot(lamda_vec, error_train, 'r')
		plt.plot(lamda_vec, error_val, 'b')
		plt.legend(['Train', 'Cross Calidation'])
		plt.xlabel('Lambda')
		plt.ylabel('Error')
		plt.xlim([0, 13])
		plt.show()

	def polyFeatures(self, x, p):
		x_ploy = np.zeros((np.size(x), p), np.float32)  # (12, 8)
		m = np.size(x)  # 12
		for i in range(m):
			for j in range(p):
				x_ploy[i, j] = x[i]**(j+1)
		return x_ploy

	def featureNormalize(self, x):
		self.mu = np.mean(x, axis=0)
		x1 = x-self.mu
		self.sigma = np.std(x1, axis=0, ddof=1)
		x = x1/self.sigma
		return [x, self.mu, self.sigma]

	def ployXandY(self):
		x = self.polyFeatures(self.x, 8)
		[x_poly, mu, sigma] = self.featureNormalize(x)

		self.x_poly = np.hstack([np.ones((x_poly.shape[0], 1)), x_poly])

		x_poly_test = self.polyFeatures(self.xtest, 8)
		x_poly_test = x_poly_test - mu
		x_poly_test = x_poly_test/sigma
		self.x_poly_test = np.hstack([np.ones((x_poly_test.shape[0], 1)), x_poly_test])

		x_poly_val = self.polyFeatures(self.xval, 8)
		x_poly_val = x_poly_val - mu
		x_poly_val = x_poly_val/sigma
		self.x_poly_val = np.hstack([np.ones((x_poly_val.shape[0], 1)), x_poly_val])
		print("Normalized Training Example 1:")
		print(self.x_poly[0, :].reshape((-1, 1)), "\n")  # 为显示美观

	def plotFit(self, x, mu, sigma, theta, p):
		x = np.arange(np.min(x) - 15, np.max(x) + 25, 0.05)
		x = x.reshape((-1, 1))
		x_poly = self.polyFeatures(x, p)
		x_poly = x_poly - mu
		x_poly = x_poly/sigma

		x_poly = np.hstack([np.ones((x_poly.shape[0], 1)), x_poly])
		plt.plot(x, x_poly.dot(theta), '--', linewidth=2)
		plt.title('Polynomial Regression Fit (lambda=0.00)')
		plt.xlabel('Change in water level (x)')
		plt.ylabel('Water flowing out of the dam (y)')
		plt.show()

	def validationCurveForLamdas(self, x, y, xval, yval, lamda_vec):
		error_train = np.zeros((len(lamda_vec), 1))
		error_val = np.zeros((len(lamda_vec), 1))
		print("Lambda\t\tTrain Error\tValidation Error\n")
		for i in range(len(lamda_vec)):
			lamda = lamda_vec[i]
			theta = self.trainLinearReg(x, y, lamda)
			error_train[i] = self.linearRegCostFunction(theta, x, y, 0)
			error_val[i] = self.linearRegCostFunction(theta, xval, yval, 0)
			print("\t\t%d\t\t\t%f\t\t%f\n" % (i, error_train[i], error_val[i]))
		return [error_train, error_val]

	def run(self):
		path = 'ex5data1.mat'
		self.loadData(path)
		self.plotXY(self.x, self.y)
		self.caculateJandGra(self.x_plus_one, self.y, 0)
		self.plotTrainingLine(self.x_plus_one, self.y)
		self.plotLinerRCurve()
		self.ployXandY()
		theta = self.trainLinearReg(self.x_poly, self.y, 0)  # 0和１都试下, 0下过拟合
		plt.scatter(self.x, self.y, marker='x', color='r')
		self.plotFit(self.x, self.mu, self.sigma, theta, 8)
		self.plotPolyNomialLearningCurve()
		self.plotValidationCurveForLambdas()


if __name__ == "__main__":
	BAV = BiasAndVariance()
	BAV.run()

