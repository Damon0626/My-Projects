# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-9 下午10:41
# @Email : wwymsn@163.com
# @Software: PyCharm

import tensorflow as tf
import scipy.io as scio
import numpy as np
import random


def loadData(path):
	data = scio.loadmat(path)
	x = data["X"]  # (5000, 400)  # 原100训练
	y = data["y"]  # (5000, 1)
	return x, y


# 处理Y,由10, 1, 2, 3, 4, 5, 6, 7, 8, 9==>0, 1, 2, 3, 4, 5, 6, 7, 8, 9
def handleYtoOne(y):
	handle_y = np.zeros((len(y), 10))
	for i in range(10):
		handle_y[len(y)//10*i:len(y)//10*(i+1), i] = 1
	return handle_y


X = tf.placeholder(tf.float32, [None, 400])
Y = tf.placeholder(tf.float32, [None, 10])

h1 = tf.Variable(tf.random_normal([400, 25]))
h2 = tf.Variable(tf.random_normal([25, 10]))

b1 = tf.Variable(tf.random_normal([25]))
b2 = tf.Variable(tf.random_normal([10]))


def neural_net(x):
	layer_1 = tf.add(tf.matmul(x, h1), b1)
	output_layer = tf.add(tf.matmul(layer_1, h2), b2)
	return output_layer


logits = neural_net(X)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))  # 交叉熵
train_op = tf.train.AdamOptimizer(0.1).minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))  # 最大值位置就是预测的结果，同真实Y比较
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # 转换为tf.float32类型

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	x1, y0 = loadData('ex4data1.mat')
	y1 = handleYtoOne(y0)
	index = random.sample([i for i in range(5000)], 4000)  # 80%training 20%testing
	train_x = x1[index, :]
	train_y = y1[index, :]
	test_x = np.delete(x1, index, 0)
	test_y = np.delete(y1, index, 0)

	for i in range(100):
		sess.run(train_op, feed_dict={X: train_x, Y: train_y})
		loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_x, Y: train_y})
		print("\r训练{}次: 损失函数{:.4f} ｜ 精度{:.4f}".format(i, loss, acc), end="")  # 精度可达94%
	print("\nTest Accuracy:%.4f%%" % (sess.run(accuracy, feed_dict={X: test_x, Y: test_y})))  # 精度89.5%
