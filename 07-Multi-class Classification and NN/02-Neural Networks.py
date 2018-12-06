# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-6 下午9:41
# @Email : wwymsn@163.com
# @Software: PyCharm

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import random

'''
两层神经网络：
输入层400个是单元；
隐藏层25个单元．
'''


class NeuralNetworks(object):
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

	def displayTestPics(self, image):
		max_val = np.max(np.abs(image))
		im = image.reshape((20, 20)).transpose()/max_val*255
		predict_result = self.predictOne(image)
		plt.xticks([])
		plt.yticks([])
		# 由于0用10表示，为了显示准确，取了余数．
		plt.title("The Prediction Result is {}!".format(np.mod(predict_result[0], 10)), color='r', fontsize=20)
		plt.imshow(im, cmap='gray')
		plt.show()

	def predictNN(self):
		x = np.hstack([np.ones((self.x.shape[0], 1)), self.x])  # 5000*401
		x1 = self.sigmoid(x.dot(self.theta1.T))  # (5000, 401)*(401, 25)

		x1_mid = np.hstack([np.ones((x1.shape[0], 1)), x1])
		x2 = self.sigmoid(x1_mid.dot(self.theta2.T))  # (5000, 26)*(26, 10)

		position = np.argmax(x2, axis=1) + 1
		accuracy = np.mean(position.reshape(5000, 1) == self.y) * 100
		print("两层神经网络准确率是：{}".format(accuracy))  # 97.52%

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def predictOne(self, image):
		x = np.hstack([np.ones((image.shape[0], 1)), image])  # 1*401
		x1 = self.sigmoid(x.dot(self.theta1.T))  # (1, 401)*(401, 25)

		x1_mid = np.hstack([np.ones((x1.shape[0], 1)), x1])
		x2 = self.sigmoid(x1_mid.dot(self.theta2.T))  # (1, 26)*(26, 10)

		position = np.argmax(x2, axis=1) + 1
		return position


	def test(self):
		test_index = random.sample([i for i in range(5000)], 5000)
		for i in test_index:
			image = self.x[i, :].reshape((1, 400))
			self.displayTestPics(image)
			s = input("Paused - press enter to continue, q to exit:")
			if s == 'q':
				break
		print("Testing is over!")


if __name__ == "__main__":
	data_path = 'ex3data1.mat'
	wight_path = 'ex3weights.mat'
	NN = NeuralNetworks()
	NN.loadData(data_path)
	NN.loadWeights(wight_path)
	NN.display100Data()
	NN.predictNN()
	NN.test()